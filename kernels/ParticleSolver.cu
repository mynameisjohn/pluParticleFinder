#include "ParticleSolver.h"
#include "ParticleFinderKernels.h"

#include <iostream>

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

// These are the default params from the centerfind testbed
ParticleFinder::Solver::impl::impl () :
    _maskRadius (3),
    _featureRadius (6),
    _minSliceCount (3),
    _maxSliceCount (5),
    _neighborRadius (5),
    _xyFactor (.0803f),
    _zFactor (0.2938f)
{}

void ParticleFinder::Solver::impl::Init (int N, int threadCount)
{
    assert (_maskRadius <= MASK_RAD_MAX);

    // Neighbor region diameter
    int diameter = 2 * _maskRadius + 1;
    cv::Size maskSize (diameter, diameter);

    // Make host mats
    cv::Mat h_Circ (maskSize, CV_32F, 0.f);
    cv::Mat h_RX (maskSize, CV_32F, 0.f);
    cv::Mat h_RY (maskSize, CV_32F, 0.f);
    cv::Mat h_R2 (maskSize, CV_32F, 0.f);

    // set up circle mask
    cv::circle (h_Circ, cv::Point (_maskRadius, _maskRadius), _maskRadius, 1.f, -1);

    // set up Rx and part of r2
    for (int y = 0; y < diameter; y++)
    {
        for (int x = 0; x < diameter; x++)
        {
            cv::Point p (x, y);
            h_RX.at<float> (p) = x + 1;
            h_RY.at<float> (p) = y + 1;
            h_R2.at<float> (p) = pow (-(float)_maskRadius + x, 2) + pow (-(float)_maskRadius + y, 2);
        }
    }

    // threshold / multiply by cicle kernel to zero values outside radius
    // (maybe we could multiply h_R2 by circ as well, but the IPP code did a threshold)
    cv::threshold (h_R2, h_R2, pow ((double)_maskRadius, 2), 1, cv::THRESH_TOZERO_INV);
    cv::multiply (h_RX, h_Circ, h_RX);
    cv::multiply (h_RY, h_Circ, h_RY);

    // For host debugging
#if SOLVER_DEVICE
    // Upload to continuous gpu mats
    _circleMask = GetContinuousGpuMat (h_Circ);
    _radXKernel = GetContinuousGpuMat (h_RX);
    _radYKernel = GetContinuousGpuMat (h_RY);
    _radSqKernel = GetContinuousGpuMat (h_R2);
#else
    h_Circ.copyTo (m_dCircleMask);
    h_RX.copyTo (m_dRadXKernel);
    h_RY.copyTo (m_dRadYKernel);
    h_R2.copyTo (m_dRadSqKernel);
#endif

    _circleKernelPtr = Floatptr((float*) _circleMask.data);
    _radXKernelPtr = Floatptr ((float*) _radXKernel.data);
    _radYKernelPtr = Floatptr ((float*) _radYKernel.data);
    _radSqKernelPtr = Floatptr ((float*) _radSqKernel.data);

    for (int i = 0; i < threadCount; i++)
    {
        _newIndicesVec.emplace_back (N * N);
        _stackToParticles.emplace_back ();
    }
}

template<int maskRadius, std::enable_if<maskRadius == 0>::type* = nullptr>
void makeParticleFromIdxWithMaskRadius (IntVec& newIndicesVec, ParticleVec& newParticlesVec, size_t newParticleCount,
    int stackNum, int sliceIdx, int N, int mR, float* lm, float* cK, float* xK, float* yK, float* sqK)
{
    assert (false);
}

template<int maskRadius, std::enable_if<maskRadius != 0>::type* = nullptr>
void makeParticleFromIdxWithMaskRadius (IntVec& newIndicesVec, ParticleVec& newParticlesVec, size_t newParticleCount, 
    int stackNum, int sliceIdx, int N, int mR, float* lm, float* cK, float* xK, float* yK, float* sqK)
{
    if (maskRadius == mR)
    {
        thrust::transform (newIndicesVec.begin (), newIndicesVec.begin () + newParticleCount, newParticlesVec.begin (),
            MakeParticleFromIdx<maskRadius> (stackNum, sliceIdx, N, lm, cK, xK, yK, sqK));
        return;
    }

    makeParticleFromIdxWithMaskRadius<maskRadius - 1> (newIndicesVec, newParticlesVec, newParticleCount,
        stackNum, sliceIdx, N, mR, lm, cK, xK, yK, sqK);
}

int ParticleFinder::Solver::impl::FindParticlesInImage (int threadNum, int stackNum, int sliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, std::vector<FoundParticle>* pParticlesInImg)
{
    // Make sure we've initialized something
    if (_radSqKernel.empty ())
        return 0;

    // Make a device vector out of the particle buffer pointer (it's contiguous)
#if SOLVER_DEVICE
    UcharPtr d_pParticleImgBufStart ((unsigned char*)d_ParticleImg.datastart);
    UcharPtr d_pParticleImgBufEnd ((unsigned char*)d_ParticleImg.dataend);
    Floatptr d_pThreshImgBuf ((float*)d_ThreshImg.data);
#else
    // For host debugging
    cv::Mat h_ThreshImg;
    d_ThreshImg.download (h_ThreshImg);
    Floatptr d_pThreshImgBuf (h_ThreshImg.ptr<float> ());

    Floatptr d_pLocalMaxImgBuf (h_ThreshImg.ptr<float> ());
    thrust::device_vector<unsigned char> d_vData ((unsigned char*)d_ParticleImg.datastart, (unsigned char*)d_ParticleImg.dataend);
    UcharVec d_ParticleImgVec = d_vData;
#endif

    const int N = d_Input.rows;

    // For each pixel in the particle image, we care if it's nonzero and if it's far enough from the edges
    // So we need its index (transformable into twoD pos) and its value, which we zip
    auto itDetectParticleBegin = thrust::make_zip_iterator (thrust::make_tuple (d_pParticleImgBufStart, thrust::counting_iterator<int> (0)));
    auto itDetectParticleEnd = thrust::make_zip_iterator (thrust::make_tuple (d_pParticleImgBufEnd, thrust::counting_iterator<int> (N * N)));

    auto& newIndicesVec = _newIndicesVec[threadNum];

    // Then, if the particle fits our criteria, we copy its index (from the counting iterator) into this vector, and discard the uchar
    auto itFirstNewParticle = thrust::make_zip_iterator (thrust::make_tuple (thrust::discard_iterator<> (), newIndicesVec.begin ()));
    auto itLastNewParticle = thrust::copy_if (itDetectParticleBegin, itDetectParticleEnd, itFirstNewParticle, IsParticleAtIdx (N, _featureRadius));
    size_t newParticleCount = itLastNewParticle - itFirstNewParticle;

    // Now transform each index into a particle by looking at the source image data surrounding the particle
    ParticleVec newParticlesVec (newParticleCount);
    makeParticleFromIdxWithMaskRadius<MASK_RAD_MAX> (newIndicesVec, newParticlesVec, newParticleCount,
        stackNum, sliceIdx, N, _maskRadius, d_pThreshImgBuf.get (), _circleKernelPtr.get (), _radXKernelPtr.get (), _radYKernelPtr.get (), _radSqKernelPtr.get ());

//    thrust::transform (newIndicesVec.begin (), newIndicesVec.begin () + newParticleCount, newParticlesVec.begin (),
//#if SOLVER_DEVICE
//        MakeParticleFromIdx (stackNum, sliceIdx, N, d_pThreshImgBuf.get (), _circleKernelPtr.get (), _radXKernelPtr.get (), _radYKernelPtr.get (), _radSqKernelPtr.get ()));
//#else
//        MakeParticleFromIdx (stackNum, sliceIdx, N, d_pThreshImgBuf, _circleKernelPtr, _radXKernelPtr, _radYKernelPtr, _radSqKernelPtr));
//#endif

    // Store new particles if requested
    if (pParticlesInImg)
    {
        // Convert all found particles to the foundparticle type
        // Reserve enough space for all previously found, though the actual may be less
        FoundParticleVec d_vRet (newParticleCount);
        pParticlesInImg->resize (newParticleCount); // (needed?)

        // We consider a particle "found" if it's intensity state is severed
        // Transform all found particles that meet the IsFoundParticle criterion into FoundParticles
        auto itFoundParticleEnd = thrust::transform (newParticlesVec.begin (), newParticlesVec.end (), d_vRet.begin (), Particle2FoundParticle ());
        auto nParticles = itFoundParticleEnd - d_vRet.begin ();

        // Download to host
        thrust::copy (d_vRet.begin (), d_vRet.end (), pParticlesInImg->begin ());
    }

    _stackToParticles[threadNum][stackNum][sliceIdx] = std::move (newParticlesVec);
    return newParticleCount;
}

std::map<int, std::vector<ParticleFinder::FoundParticle>> ParticleFinder::Solver::impl::LinkFoundParticles ()
{
    std::map<int, std::vector<ParticleFinder::FoundParticle>> ret;
    std::map<int, std::map<int, ParticleVec>> combinedParticles;
    for (auto& perThreadParticles : _stackToParticles)
    {
        for (auto& stackToSlices : perThreadParticles)
        {
            for (auto it = stackToSlices.second.begin (); it != stackToSlices.second.end (); )
            {
                combinedParticles[stackToSlices.first][it->first] = std::move (it->second);
                it = stackToSlices.second.erase (it);
            }
        }
    }

    for (auto& itParticles : combinedParticles)
    {
        auto& sliceToParticles = itParticles.second;

        auto itPrev = sliceToParticles.begin ();
        auto itCur = std::next (itPrev);
        float max_r2_dev = _neighborRadius * _neighborRadius;

        SeverParticle severParticle;
        ShouldSeverParticle shouldSeverParticle{ _minSliceCount, _maxSliceCount };
        size_t count = thrust::count_if (itPrev->second.begin (), itPrev->second.end (), IsFoundParticle ());
        int sliceIdx = 1;
        for (; itCur != sliceToParticles.end (); ++itPrev, ++itCur, sliceIdx++)
        {
            auto& prevParticleVec = itPrev->second;
            auto& curParticleVec = itCur->second;

            int numParticlePairs = curParticleVec.size () * prevParticleVec.size ();
            int numPrev = prevParticleVec.size ();

            auto itDetectParticleBegin = thrust::counting_iterator<int> (0);
            auto itDetectParticleEnd = itDetectParticleBegin + numParticlePairs;

#if SOLVER_DEVICE
            CheckParticleRadius checkParticleRadius{ prevParticleVec.data ().get (), curParticleVec.data ().get (), numPrev, max_r2_dev };
            AttachParticle attachParticle{ prevParticleVec.data ().get (), curParticleVec.data ().get (), numPrev };
#else
            CheckParticleRadius checkParticleRadius{ prevParticleVec.data (), curParticleVec.data (), numPrev, max_r2_dev };
            AttachParticle attachParticle{ prevParticleVec.data (), curParticleVec.data (), numPrev };
#endif

            thrust::transform_if (thrust_exec, itDetectParticleBegin, itDetectParticleEnd, thrust::discard_iterator<> (), attachParticle, checkParticleRadius);
            thrust::transform_if (curParticleVec.begin (), curParticleVec.end (), thrust::discard_iterator<> (), severParticle, shouldSeverParticle);

            size_t particleInSlice = thrust::count_if (curParticleVec.begin (), curParticleVec.end (), IsFoundParticle ());

#if DEBUG
            std::cout << particleInSlice << " particles linked in stack " << itParticles.first << ", slice " << sliceIdx << std::endl;
#endif

            count += particleInSlice;
        }

        // init found particles
        for (auto& particles : sliceToParticles)
            thrust::transform_if (particles.second.begin (), particles.second.end (), thrust::discard_iterator<> (), initFoundParticles (_xyFactor), IsFoundParticle ());

        // reduce children into found particles
        for (auto& particles : sliceToParticles)
            thrust::transform_if (particles.second.begin (), particles.second.end (), thrust::discard_iterator<> (), computeAverageSum (_xyFactor), IsNotFoundParticle ());

        // average positions
        for (auto& particles : sliceToParticles)
            thrust::for_each (particles.second.begin (), particles.second.end (), averageParticlePositions (_xyFactor, _zFactor));

        // remove children
        FilterFoundParticlesBySliceCount filterFoundParticlesBySliceCount{ _minSliceCount };
        for (auto& particles : sliceToParticles)
        {
            auto it = thrust::remove_if (particles.second.begin (), particles.second.end (), filterFoundParticlesBySliceCount);
            particles.second.erase (it, particles.second.end ());
        }

        count = 0;
        for (auto& particles : sliceToParticles)
            count += particles.second.size ();

        float boundary_r{ 0.100000001f };
        float minX{ std::numeric_limits<float>::max () };
        float minY{ std::numeric_limits<float>::max () };
        float minZ{ std::numeric_limits<float>::max () };
        float maxX{ std::numeric_limits<float>::min () };
        float maxY{ std::numeric_limits<float>::min () };
        float maxZ{ std::numeric_limits<float>::min () };

        for (auto& particles : sliceToParticles)
        {
            auto X = thrust::minmax_element (particles.second.begin (), particles.second.end (), MinVectorElement<0> ());
            minX = std::min (minX, ((Particle)*X.first).x + boundary_r);
            maxX = std::max (maxX, ((Particle)*X.second).x - boundary_r);

            auto Y = thrust::minmax_element (particles.second.begin (), particles.second.end (), MinVectorElement<1> ());
            minY = std::min (minY, ((Particle)*Y.first).y + boundary_r);
            maxY = std::max (maxY, ((Particle)*Y.second).y - boundary_r);

            auto Z = thrust::minmax_element (particles.second.begin (), particles.second.end (), MinVectorElement<2> ());
            minZ = std::min (minZ, ((Particle)*Z.first).z + boundary_r);
            maxZ = std::max (maxZ, ((Particle)*Z.second).z - boundary_r);
        }

        CheckParticleBoundaries checkParticleBoundaries{ minX, minY, minZ, maxX, maxY, maxZ };
        for (auto& particles : sliceToParticles)
        {
            auto it = thrust::remove_if (particles.second.begin (), particles.second.end (), checkParticleBoundaries);
            particles.second.erase (it, particles.second.end ());
        }

        count = 0;
        for (auto& particles : sliceToParticles)
            count += particles.second.size ();

        auto& hostParticleVec = ret[itParticles.first];
        hostParticleVec.resize (count);
        auto itHostCopy = hostParticleVec.begin ();
        for (auto& particles : sliceToParticles)
        {
            if (particles.second.empty ())
                continue;

            FoundParticleVec foundParticles (particles.second.size ());
            thrust::transform (particles.second.begin (), particles.second.end (), foundParticles.begin (), Particle2FoundParticle ());
            itHostCopy = thrust::copy (foundParticles.begin (), foundParticles.end (), itHostCopy);
        }

#if DEBUG
        std::cout << count << " total particles found in stack " << itParticles.first << std::endl;
#endif
    }

#if DEBUG
    std::cout << "Linking complete" << std::endl;
#endif

    return ret;
}