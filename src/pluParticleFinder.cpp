#include "ParticleFinder.h"

#include <FreeImage.h>
#include <algorithm>
#include <set>
#include <fstream>

bool ParticleFinder::Initialize(std::list<std::string> liStackPaths, int nStartOfStack, int nEndOfStack, bool bDoUserInput /*= false*/)
{
    // For each tiff stack
    int nSlices(0);
#if DEBUG
    std::cout << "Starting to load slices..." << std::endl;
#endif

    // Load the images - this could be done
    // concurrently to the CenterFind DSP
    int ixStack (0);
    for (const std::string& strStackPath : liStackPaths)
    {
        // Attempt to open multibitmap
        FIMULTIBITMAP * FI_Input = FreeImage_OpenMultiBitmap(FIF_TIFF, strStackPath.c_str(), 0, 1, 1, TIFF_DEFAULT);
        if (FI_Input == nullptr)
        {
#if DEBUG
            std::cout << "Error loading stack " << strStackPath << std::endl;
            continue;
#endif
        }
#if DEBUG
        std::cout << "Loading stack " << strStackPath << std::endl;
#endif

        // Read in images, create data

        int nImages = FreeImage_GetPageCount(FI_Input);
        nImages = std::min(nEndOfStack, FreeImage_GetPageCount(FI_Input));
        for (int ixSlice = std::max(nStartOfStack, 0); ixSlice < nImages; ixSlice++)
        {
            if (FIBITMAP * image = FreeImage_LockPage(FI_Input, ixSlice - 1))
            {
                // Create 24-bit RGB image, initialize to zero (this could be a member)
                int nWidth = FreeImage_GetWidth(image);
                int nHeight = FreeImage_GetHeight(image);
                cv::Size imgSize(nWidth, nHeight);
                cv::Mat m = cv::Mat::zeros(imgSize, CV_8UC3);

                // If we haven't yet created our internal mats,
                // do so now (and assume all will be of same dimension...)
                if (_inputImages.empty())
                {
                    // Initialize all other mats to zero on device (may be superfluous)
                    _filteredImg = GpuMat(imgSize, CV_32F, 0.f);
                    _dilatedImg = GpuMat(imgSize, CV_32F, 0.f);
                    _scratchImg = GpuMat(imgSize, CV_32F, 0.f);
                    _localMaxImg = GpuMat(imgSize, CV_32F, 0.f);
                    _scratchImg = GpuMat(imgSize, CV_32F, 0.f);

                    // It's in our best interest to ensure these are continuous
                    cv::cuda::createContinuous(imgSize, CV_32F, _threshImg);
                    cv::cuda::createContinuous(imgSize, CV_8U, _particleImg);
                    _threshImg.setTo(0);
                    _particleImg.setTo(0);
                }

                // Convert FIBITMAP to 24 bit RGB, store inside cv::Mat's buffer
                FreeImage_ConvertToRawBits(m.data, image, m.step, 24, 0xFF, 0xFF, 0xFF, true);

                // Upload to device (should this get a separate buffer?)
                GpuMat d_InputImg;
                d_InputImg.upload(m);

                // Convert to greyscale float, store in our input buffer
                cv::cuda::cvtColor(d_InputImg, _scratchImg, cv::COLOR_RGB2GRAY);
                _scratchImg.convertTo(d_InputImg, CV_32F, 1. / 0xFF);
                _inputImages.push_back(d_InputImg);
                _imageToStackFrame[nSlices++] = std::make_pair (ixStack + 1, ixSlice);
                // Inc slice count
            }
            else
            {
#if DEBUG
                std::cout << "Error loading slice " << ixSlice << " of stack " << strStackPath << std::endl;
#endif
            }

#if DEBUG
            // Print something out every 10 slices
            if (ixSlice % 10 == 0)
                std::cout << "On image " << ixSlice << " of stack " << strStackPath << "..." << std::endl;
#endif
        }

        // Close multibitmap
        FreeImage_CloseMultiBitmap(FI_Input);
        ixStack++;
    }

    if (!_inputImages.empty()) 
    {
        _solver.Init ();

        if (bDoUserInput)
            getUserInput (_inputImages.front ());

        return true;
    }

    return false;
}

std::vector<ParticleFinder::FoundParticle>  ParticleFinder::launchTask (std::shared_ptr<AsyncParticleFindingTask> spParticleFindingTask)
{
    // Launch thread
    LockMutex (spParticleFindingTask->muData);
    m_spParticleFindingTask = spParticleFindingTask;
    m_fuParticleFindingTask = std::async (std::launch::async, [this, spParticleFindingTask]()
        {
            // Set DSP params
            SetGaussianRadius (m_spParticleFindingTask->nGaussianRadius);
            SetFWHM (m_spParticleFindingTask->fHWHM);
            SetDilationRadius (m_spParticleFindingTask->nDilationRadius);
            SetParticleThreshold (m_spParticleFindingTask->fParticleThreshold);
            GetSolver ()->SetMaskRadius (m_spParticleFindingTask->nMaskRadius);
            GetSolver ()->SetFeatureRadius (m_spParticleFindingTask->nFeatureRadius);
            GetSolver ()->SetMinSliceCount (m_spParticleFindingTask->nMinSliceCount);
            GetSolver ()->SetMaxSliceCount (m_spParticleFindingTask->nMaxSliceCount);
            GetSolver ()->SetNeighborRadius (m_spParticleFindingTask->nNeighborRadius);
            GetSolver ()->Init ();

            // Make sure this is empty
            m_spParticleFindingTask->mapFoundSliceToParticles.clear ();

            for (size_t i = 0; i < _inputImages.size (); i++)
            {
                {    // Check to see if we're cancelled
                    LockMutex (m_spParticleFindingTask->muData);
                    if (m_spParticleFindingTask->bCancel)
                    {
                        // Reset solver (clears found particles) and flag done
                        GetSolver ()->ResetLinking ();
                        m_spParticleFindingTask->bIsDone = true;
                        return;
                    }
                }

                // Find particles in image
                doDSPAndFindParticlesInImg (0, (int)i, _inputImages[i]);

                // Update task with found particles (we have to look at an earlier slice,
                // as the particle center won't be in the slice we just processed)
                if (i >= m_spParticleFindingTask->nMaxSliceCount)
                {
                    int ixFindParticles = (int)(i - m_spParticleFindingTask->nMaxSliceCount);
                    {
                        LockMutex (m_spParticleFindingTask->muData);
                        m_spParticleFindingTask->nLastImageProcessed = ixFindParticles;
                        // m_spParticleFindingTask->mapFoundSliceToParticles[ixFindParticles] = m_Solver.GetFoundParticlesInSlice (ixFindParticles);
                    }
                }

            }

            {    // Verdun
                std::lock_guard<std::mutex> lg (m_spParticleFindingTask->muData);
                m_spParticleFindingTask->bIsDone = true;
            }

            return;
        });

    return{};
}

int ParticleFinder::doDSPAndFindParticlesInImg(int stackNum, int ixSlice, GpuMat d_Input, std::vector<FoundParticle> * pFoundParticles /*= nullptr*/, bool bResetKernels /*= false*/)
{
    if ( bResetKernels || _circleFilter.empty() || _dilationKernel.empty() )
    {
#if DEBUG
        std::cout << "Constructing DSP kernels" << std::endl;
#endif

        int nBPDiameter = 2 * _gaussFiltRadius + 1;
        const cv::Size bpFilterSize (nBPDiameter, nBPDiameter);

        // Create circle image
        cv::Mat h_Circle (bpFilterSize, CV_32F, 0.f); // cv::Mat::zeros (cv::Size (nBPDiameter, nBPDiameter), CV_32F);
        cv::circle( h_Circle, cv::Point ( _gaussFiltRadius, _gaussFiltRadius ), _gaussFiltRadius, 1.f, -1 );
        _circleFilter = cv::cuda::createLinearFilter( CV_32F, CV_32F, h_Circle );

        // Create Gaussian Filter and normalization scale
        const double dSigma = (double)(_HWHM / 0.8325546) / 2;
        _gaussFilter = cv::cuda::createGaussianFilter( CV_32F, CV_32F, bpFilterSize, dSigma );

        // Create dilation mask
        int nDilationDiameter = 2 * _dilationRadius + 1;
        cv::Mat h_Dilation = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( nDilationDiameter, nDilationDiameter ) );

        // Create dilation kernel from host kernel (only single byte supported? why nVidia why)
        _dilationKernel = cv::cuda::createMorphologyFilter( cv::MORPH_DILATE, CV_32F, h_Dilation );
    }

#if DEBUG
    std::cout << "Finding particles in slice " << ixSlice << std::endl;
#endif

    RemapImage (d_Input, 0, 100);
    
    // Apply low/high pass filters (gaussian and circle kernels, resp.)
    _gaussFilter->apply( d_Input, _filteredImg );
    //ShowImage( m_dFilteredImg );

    _circleFilter->apply( d_Input, _scratchImg );
    //ShowImage( m_dTmpImg );
    double dBandPassScale = 1 / (3 * pow (_gaussFiltRadius, 2));
    _scratchImg.convertTo( _scratchImg, CV_32F, dBandPassScale);

    // subtract tmp from bandpass to get filtered output
    cv::cuda::subtract( _filteredImg, _scratchImg, _filteredImg );
    //ShowImage( m_dFilteredImg );

    // Any negative values become 0
    cv::cuda::threshold( _filteredImg, _filteredImg, 0, 1, cv::THRESH_TOZERO );
    //ShowImage( m_dFilteredImg );

    // Remap this from 0 to 100, so we have a normalized
    // range of particle intenstiy thresholds
    RemapImage( _filteredImg, 0, 100 );

    // Reset thresh to base value
    _threshImg.setTo( cv::Scalar( _particleThreshold ) );

    // Store max of (m_fParticleThreshold, m_dFilteredImg) in m_dThreshImg
    cv::cuda::max( _threshImg, _filteredImg, _threshImg );
    //ShowImage( m_dThreshImg );

    // Initialize dilated image to threshold value
    _dilatedImg.setTo( cv::Scalar( _particleThreshold ) );

    // Apply dilation
    _dilationKernel->apply( _threshImg, _dilatedImg );
    //( m_dDilatedImg );

    // Subtract filtered image from dilated (negates pixels that are not local maxima)
    cv::cuda::subtract( _filteredImg, _dilatedImg, _localMaxImg );

    // Exponentiate m_dLocalMaxImg (sends negative values to low numbers, lm to 0)
    cv::cuda::exp( _localMaxImg, _localMaxImg );

    // Threshold exponentiated image - we are left with only local maxima as nonzero pixels
    //ShowImage( m_dLocalMaxImg );
    constexpr double epsilon (0.0000001);
    cv::cuda::threshold( _localMaxImg, _localMaxImg, 1.0 - epsilon, 1, cv::THRESH_BINARY );
    
    // Cast to uchar, store as particle image (values are still 0 or 1, so no scale needed)
    _localMaxImg.convertTo( _particleImg, CV_8U );

    return _solver.FindParticlesInImage(stackNum, ixSlice, d_Input, _filteredImg, _threshImg, _particleImg, pFoundParticles );
}

// Returns empty if being launched asynchronously
std::vector<ParticleFinder::FoundParticle> ParticleFinder::Execute (bool linkParticles, std::shared_ptr<AsyncParticleFindingTask> spParticleFindingTask /*= nullptr*/)
{
    // Construct filters now
    // make sure that we have valid params)
    if (_inputImages.empty () || _dilationRadius == 0 || _gaussFiltRadius == 0)
        return{};

    // If we have a task going, cancel it now
    if (m_spParticleFindingTask)
    {
        cancelTask ();
    }

    // If an async task was passed in, we return empty
    // and kick the task off with the supplied params
    if (spParticleFindingTask)
    {
        return launchTask (spParticleFindingTask);
    }
    else
    {
        std::vector<FoundParticle> particlesInImg, * pParticlesInImg{ nullptr };
        std::ofstream outputFile;
        if (!_outputFile2D.empty())
        {
            outputFile.open (_outputFile2D, std::ios::out);
            outputFile.setf (std::ios::fixed);
            outputFile.precision (1);
            pParticlesInImg = &particlesInImg;
        }

        // Otherwise we do the DSP with our current params and return found particles
        for (size_t i = 0; i < _inputImages.size (); i++)
        {
            int width = _inputImages[0].cols;
            doDSPAndFindParticlesInImg (_imageToStackFrame[i].first , (int)i, _inputImages[i], pParticlesInImg);

            if (outputFile.is_open())
            {
                for (int p = 0; p < particlesInImg.size (); p++)
                {
                    outputFile << _imageToStackFrame[i].first << '\t';
                    outputFile << _imageToStackFrame[i].second << '\t';
                    outputFile << particlesInImg[p].fPosX << '\t';
                    outputFile << particlesInImg[p].fPosY << '\t';
                    outputFile << particlesInImg[p].fIntensity << '\t';
                    outputFile << sqrt (particlesInImg[p].fR2) << std::endl;
                }
            }
        }

        if (linkParticles)
        {
#if DEBUG
            std::cout << "Linking particles" << std::endl;
#endif

            std::ofstream particlePosFile (_outputFileXYZT, std::ios::out);
            for (auto& it : _solver.LinkFoundParticles ())
            {
                for (auto& particle : it.second)
                {
                    particlePosFile << it.first << '\t';
                    particlePosFile << particle.fPosX << '\t';
                    particlePosFile << particle.fPosY << '\t';
                    particlePosFile << particle.fPosZ << std::endl;
                }
            }
        }

        // Return all found particles
        // return m_Solver.GetFoundParticles ();
        return {};
    }
}

// Blocking call to cancel particle finding task
void ParticleFinder::cancelTask ()
{
    if (m_spParticleFindingTask)
    {
        {
            LockMutex (m_spParticleFindingTask->muData);
            m_spParticleFindingTask->bCancel = true;
        }

        // Wait till it's done and clear shared pointer
        m_fuParticleFindingTask.get ();
        m_spParticleFindingTask = nullptr;
    }
}

// Quick function to create continuous gpumat from existing host mat
cv::cuda::GpuMat GetContinuousGpuMat( cv::Mat& m )
{
    cv::cuda::GpuMat ret = cv::cuda::createContinuous( m.size(), m.type() );
    ret.upload( m );
    return ret;
}

void RemapImage( cv::cuda::GpuMat& img, double m1, double M1 )
{
    double m0( 1 ), M0( 2 );
    cv::cuda::minMax( img, &m0, &M0 );
    double a = ( M1 - m1 ) / ( M0 - m0 );
    double b = m1 - a * m0;
    img.convertTo( img, img.type(), a, b );
}

void RemapImage( cv::Mat& img, double m1, double M1 )
{
    double m0( 1 ), M0( 2 );
    cv::minMaxLoc( img, &m0, &M0 );
    double a = ( M1 - m1 ) / ( M0 - m0 );
    double b = m1 - a * m0;
    img.convertTo( img, img.type(), a, b );
}

void ShowImage( cv::cuda::GpuMat& img )
{
    cv::cuda::GpuMat disp;
    img.convertTo( disp, CV_32F );
    RemapImage( disp, 0., 1. );
    std::string winName( "disp" );
    cv::namedWindow( winName, cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE );
    cv::imshow( winName, disp );
    cv::waitKey();
}

void ShowImage( cv::Mat& img )
{
    cv::Mat disp;
    img.convertTo( disp, CV_32F );
    RemapImage( disp, 0., 1. );
    std::string winName( "disp" );
    cv::namedWindow( winName, cv::WINDOW_AUTOSIZE );
    cv::imshow( winName, disp );
    cv::waitKey();
}