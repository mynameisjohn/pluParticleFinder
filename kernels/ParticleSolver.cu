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

void ParticleFinder::Solver::impl::Init ()
{
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
}

int ParticleFinder::Solver::impl::FindParticlesInImage (int stackNum, int nSliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, std::vector<FoundParticle>* pParticlesInImg)
{
    // Make sure we've initialized something
    if (_radSqKernel.empty ())
        return 0;

    // Make a device vector out of the particle buffer pointer (it's contiguous)
#if SOLVER_DEVICE
    UcharPtr d_pParticleImgBufStart ((unsigned char*)d_ParticleImg.datastart);
    UcharPtr d_pParticleImgBufEnd ((unsigned char*)d_ParticleImg.dataend);
    UcharVec d_ParticleImgVec (d_pParticleImgBufStart, d_pParticleImgBufEnd);
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

    // Get pointers to our kernels (can we cache these + other operators?)
    Floatptr d_pCirleKernel ((float*)_circleMask.data);
    Floatptr d_pRxKernel ((float*)_radXKernel.data);
    Floatptr d_pRyKernel ((float*)_radYKernel.data);
    Floatptr d_pR2Kernel ((float*)_radSqKernel.data);

    // For each pixel in the particle image, we care if it's nonzero and if it's far enough from the edges
    // So we need its index (transformable into twoD pos) and its value, which we zip
    auto itDetectParticleBegin = thrust::make_zip_iterator (thrust::make_tuple (d_ParticleImgVec.begin (), thrust::counting_iterator<int> (0)));
    auto itDetectParticleEnd = thrust::make_zip_iterator (thrust::make_tuple (d_ParticleImgVec.end (), thrust::counting_iterator<int> (N * N)));

    // Then, if the particle fits our criteria, we copy its index (from the counting iterator) into this vector, and discard the uchar
    IntVec d_NewParticleIndicesVec (N * N);
    auto itFirstNewParticle = thrust::make_zip_iterator (thrust::make_tuple (thrust::discard_iterator<> (), d_NewParticleIndicesVec.begin ()));
    auto itLastNewParticle = thrust::copy_if (itDetectParticleBegin, itDetectParticleEnd, itFirstNewParticle, IsParticleAtIdx (N, _featureRadius));
    size_t newParticleCount = itLastNewParticle - itFirstNewParticle;

    // Now transform each index into a particle by looking at the source image data surrounding the particle
    ParticleVec d_NewParticleVec (newParticleCount);
    thrust::transform (d_NewParticleIndicesVec.begin (), d_NewParticleIndicesVec.begin () + newParticleCount, d_NewParticleVec.begin (),
#if SOLVER_DEVICE
        MakeParticleFromIdx (stackNum, nSliceIdx, N, _maskRadius, d_pThreshImgBuf.get (), d_pCirleKernel.get (), d_pRxKernel.get (), d_pRyKernel.get (), d_pR2Kernel.get ()));
#else
        MakeParticleFromIdx (stackNum, nSliceIdx, N, m_uMaskRadius, d_pThreshImgBuf, d_pCirleKernel, d_pRxKernel, d_pRyKernel, d_pR2Kernel));
#endif

    // Store new particles if requested
    if (pParticlesInImg)
    {
        // Convert all found particles to the foundparticle type
        // Reserve enough space for all previously found, though the actual may be less
        FoundParticleVec d_vRet (newParticleCount);
        pParticlesInImg->resize (newParticleCount); // (needed?)

        // We consider a particle "found" if it's intensity state is severed
        // Transform all found particles that meet the IsFoundParticle criterion into FoundParticles
        auto itFoundParticleEnd = thrust::transform (d_NewParticleVec.begin (), d_NewParticleVec.end (), d_vRet.begin (), Particle2FoundParticle ());
        auto nParticles = itFoundParticleEnd - d_vRet.begin ();

        // Download to host
        thrust::copy (d_vRet.begin (), d_vRet.end (), pParticlesInImg->begin ());
    }

    _stackToParticles[stackNum].emplace_back (std::move (d_NewParticleVec));
    return newParticleCount;
}

std::map<int, std::vector<ParticleFinder::FoundParticle>> ParticleFinder::Solver::impl::LinkFoundParticles ()
{
    std::map<int, std::vector<ParticleFinder::FoundParticle>> ret;

    for (auto& itParticles : _stackToParticles)
    {
        auto& m_vParticles = itParticles.second;

        auto itPrev = m_vParticles.begin ();
        auto itCur = std::next (itPrev);
        float max_r2_dev = _neighborRadius * _neighborRadius;

        SeverParticle severParticle;
        ShouldSeverParticle shouldSeverParticle{ _minSliceCount, _maxSliceCount };
        size_t count = thrust::count_if (itPrev->begin (), itPrev->end (), IsFoundParticle ());
        int sliceIdx = 1;
        for (; itCur != m_vParticles.end (); ++itPrev, ++itCur, sliceIdx++)
        {
            int numParticlePairs = itCur->size () * itPrev->size ();
            int numPrev = itPrev->size ();

            auto itDetectParticleBegin = thrust::counting_iterator<int> (0);
            auto itDetectParticleEnd = itDetectParticleBegin + numParticlePairs;

#if SOLVER_DEVICE
            CheckParticleRadius checkParticleRadius{ itPrev->data ().get (), itCur->data ().get (), numPrev, max_r2_dev };
            AttachParticle attachParticle{ itPrev->data ().get (), itCur->data ().get (), numPrev };
#else
            CheckParticleRadius checkParticleRadius{ itPrev->data (), itCur->data (), numPrev, max_r2_dev };
            AttachParticle attachParticle{ itPrev->data (), itCur->data (), numPrev };
#endif

            thrust::transform_if (thrust_exec, itDetectParticleBegin, itDetectParticleEnd, thrust::discard_iterator<> (), attachParticle, checkParticleRadius);
            thrust::transform_if (itCur->begin (), itCur->end (), thrust::discard_iterator<> (), severParticle, shouldSeverParticle);

            size_t particleInSlice = thrust::count_if (itCur->begin (), itCur->end (), IsFoundParticle ());

#if DEBUG
            std::cout << particleInSlice << " particles linked in stack " << itParticles.first << ", slice " << sliceIdx << std::endl;
#endif

            count += particleInSlice;
        }

        // init found particles
        for (auto& particles : m_vParticles)
            thrust::transform_if (particles.begin (), particles.end (), thrust::discard_iterator<> (), initFoundParticles (_xyFactor), IsFoundParticle ());

        // reduce children into found particles
        for (auto& particles : m_vParticles)
            thrust::transform_if (particles.begin (), particles.end (), thrust::discard_iterator<> (), computeAverageSum (_xyFactor), IsNotFoundParticle ());

        // average positions
        for (auto& particles : m_vParticles)
            thrust::for_each (particles.begin (), particles.end (), averageParticlePositions (_xyFactor, _zFactor));

        // remove children
        FilterParticlesBySliceCount filterParticlesBySliceCount{ _minSliceCount };
        for (auto& particles : m_vParticles)
        {
            auto it = thrust::remove_if (particles.begin (), particles.end (), IsNotFoundParticle ());
            particles.erase (it, particles.end ());
            it = thrust::remove_if (particles.begin (), particles.end (), filterParticlesBySliceCount);
            particles.erase (it, particles.end ());
        }

        count = 0;
        for (auto& particles : m_vParticles)
            count += particles.size ();

        float boundary_r{ 0.100000001f };
        float minX{ std::numeric_limits<float>::max () };
        float minY{ std::numeric_limits<float>::max () };
        float minZ{ std::numeric_limits<float>::max () };
        float maxX{ std::numeric_limits<float>::min () };
        float maxY{ std::numeric_limits<float>::min () };
        float maxZ{ std::numeric_limits<float>::min () };

        for (auto& particles : m_vParticles)
        {
            auto it = thrust::remove_if (particles.begin (), particles.end (), IsNotFoundParticle ());
            particles.erase (it, particles.end ());

            auto X = thrust::minmax_element (particles.begin (), particles.end (), MinVectorElement<0> ());
            minX = std::min (minX, ((Particle)*X.first).x + boundary_r);
            maxX = std::max (maxX, ((Particle)*X.second).x - boundary_r);

            auto Y = thrust::minmax_element (particles.begin (), particles.end (), MinVectorElement<1> ());
            minY = std::min (minY, ((Particle)*Y.first).y + boundary_r);
            maxY = std::max (maxY, ((Particle)*Y.second).y - boundary_r);

            auto Z = thrust::minmax_element (particles.begin (), particles.end (), MinVectorElement<2> ());
            minZ = std::min (minZ, ((Particle)*Z.first).z + boundary_r);
            maxZ = std::max (maxZ, ((Particle)*Z.second).z - boundary_r);
        }

        CheckParticleBoundaries checkParticleBoundaries{ minX, minY, minZ, maxX, maxY, maxZ };
        for (auto& particles : m_vParticles)
        {
            auto it = thrust::remove_if (particles.begin (), particles.end (), checkParticleBoundaries);
            particles.erase (it, particles.end ());
        }

        count = 0;
        for (auto& particles : m_vParticles)
            count += particles.size ();

        auto& hostParticleVec = ret[itParticles.first];
        hostParticleVec.resize (count);
        auto itHostCopy = hostParticleVec.begin ();
        for (auto& particles : m_vParticles)
        {
            if (particles.empty ())
                continue;

            FoundParticleVec foundParticles (particles.size ());
            thrust::transform (particles.begin (), particles.end (), foundParticles.begin (), Particle2FoundParticle ());
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