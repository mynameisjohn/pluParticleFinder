#pragma once

#include <memory>
#include <list>
#include <future>
#include <opencv2/opencv.hpp>

#define LockMutex(_mu) std::lock_guard<std::mutex> _muLG(_mu)
#define LockMutexWithCode(_mu, _code) std::lock_guard<std::mutex> _muLG(_mu); {_code;}

// Particle Finder class
// Does all DSP, delegates
// particle position solving / linking
class ParticleFinder
{
public:
    // Useful typedef
    using GpuMat = cv::cuda::GpuMat;

    // FoundParticle struct
    // Represents particle 3D position, intensity, and radius
    struct FoundParticle
    {
        float fPosX;
        float fPosY;
        float fPosZ;
        float fIntensity;
        float fR2;
    };

    // Solver class
    // Find particles in 2D images and optionally links 
    // particles in 3D across several images. 
    // Implemented in .cu/cuh files because it runs some thrust/CUDA code
    class Solver
    {
        // we have a separate impl class performing the thrust/CUDA code
        struct impl;
        std::unique_ptr<Solver::impl> _solverImpl;

    public:
        Solver();
        ~Solver ();

        // Initializes kernels and internal data structures
        // (after params have been set)
        void Init (int N, int numThreads);

        // Resets the linking state
        void ResetLinking();
        
        // Find 2D particles in a given image, return the count of particles found
        // (optionally assign particlesInImg to get all particles in this image)
        int FindParticlesInImage(int threadNum,
                                 int stackNum, 
                                 int sliceIdx,
                                 GpuMat input, 
                                 GpuMat filteredImage, 
                                 GpuMat threshImg, 
                                 GpuMat particleImg,
                                 std::vector<FoundParticle> * particlesInImg = nullptr );

        // Link 2D particle positions across slices, computing 3D particle positions
        // as well as their radius / intensity. 
        // Returns a map of stack index to found particle
        std::map<int, std::vector<FoundParticle>> LinkFoundParticles ();

        // Getters and setters for the solver params (implementd by solver::impl)

        // 2D particle finding parameters
        int GetMaskRadius() const;
        void SetMaskRadius( int mR );
        int GetFeatureRadius() const;
        void SetFeatureRadius( int fR );
        
        // 3D linking params
        int GetMinSliceCount() const;
        void SetMinSliceCount( int nMinSC );        
        int GetMaxSliceCount() const;
        void SetMaxSliceCount( int nMaxSC );
        float GetNeighborRadius() const;
        void SetNeighborRadius( float nR );
        void SetXYFactor (float xyFactor);
        float GetXYFactor () const;
        void SetZFactor (float zFactor);
        float GetZFactor () const;
    } _solver;

    ///////////////////////////////////////////////////////////////////

    std::vector<GpuMat> _inputImages;    // Buffer used for input
    std::map<int, std::pair<int, int>> _imageToStackFrame; // used to map input images to stacks / frams

    // Internal buffers - these are of the same
    // dimensions, but may be different data types
    struct procData
    {
        GpuMat filteredImg;    // Post Gaussian Bandpass
        GpuMat dilatedImg;    // Post dilation
        GpuMat localMaxImg;    // Local Max image
        GpuMat particleImg;    // Particle Image
        GpuMat threshImg;    // Thresholded Particle image
        GpuMat scratchImg;        // Scrach buffer image
        cv::Ptr<cv::cuda::Filter> gaussFilter;        // Gaussian Filter
        cv::Ptr<cv::cuda::Filter> circleFilter;        // Circle Filter
        cv::Ptr<cv::cuda::Filter> dilationKernel;    // Dilation Filter
        cv::cuda::Stream stream;
    };
    std::vector<procData> _procData;

    ///////////////////////////////////////////////////////////////////
    // DSP parameters
    // These are the default testbed params
    int _gaussFiltRadius{ 6 };         // Radius of Gaussian (low-pass) Filter
    int _dilationRadius{ 3 };          // Radius of dilation operation
    float _HWHM{ 4.f };                // Half-Width at Half Maximum (describes gaussian kernel)
    float _particleThreshold{ 5.f };   // Threshold intensity for particles in image

    // This actual detects particles in the input image
    // the result is a contiguous black and white
    // image that can be used in particle detection
    int doDSPAndFindParticlesInImg( int stackNum, int ixSlice, GpuMat d_Input, std::vector<FoundParticle> * pFoundParticles = nullptr );

    void getUserInput( GpuMat d_Input );
    
    std::string _outputFile2D;
    std::string _outputFileXYZT;

public:
    // Initializes DSP params to default testbed params
    ParticleFinder () = default;
    ~ParticleFinder () { cancelTask (); }

    // Simple getter for solver to make it accessible
    Solver * GetSolver() { return &_solver; }

    // The other way is to use these setters
    inline void SetGaussianRadius (int nGaussianRadius) { _gaussFiltRadius = nGaussianRadius; }
    inline void SetDilationRadius (int nDilationRadius) { _dilationRadius = nDilationRadius; }
    inline void SetFWHM (int fHWHM) { _HWHM = fHWHM; }
    inline void SetParticleThreshold (int fParticleThreshold) { _particleThreshold = fParticleThreshold; }

    inline int GetGaussianRadius () const { return _gaussFiltRadius; }
    inline int GetDilationRadius () const { return _gaussFiltRadius; }
    inline float GetFWHM () const { return _HWHM; }
    inline float GetParticleThreshold () const { return _particleThreshold; }

    void SetOutputFile2D (std::string outputFile2D) { _outputFile2D = outputFile2D; }
    void SetOutputFileXYZT (std::string outputFileXYZT) { _outputFileXYZT = outputFileXYZT; }

    bool Initialize(std::list<std::string> liStackPaths, int nStartOfStack, int nEndOfStack, bool bDoUserInput = false);
    void ResetKernels ();

    size_t GetNumImages () const { return _inputImages.size (); }
    cv::Size GetImageDimensions () { return _inputImages.empty () ? cv::Size() : _inputImages.front ().size (); }

    //! \todo What's the plan for these async tasks?
    struct AsyncParticleFindingTask {
        std::mutex muData;
        bool bCancel;
        bool bIsDone;
        ParticleFinder * pParticleFinder;
        int nLastImageProcessed;
        int nGaussianRadius;
        float fHWHM;
        int nDilationRadius;
        float fParticleThreshold;
        int nMaskRadius;
        int nFeatureRadius;
        int nMinSliceCount;
        int nMaxSliceCount;
        int nNeighborRadius;
        std::map<int, std::vector<FoundParticle>> mapFoundSliceToParticles;
    };
    std::vector<FoundParticle> Execute(bool linkParticles, std::shared_ptr<AsyncParticleFindingTask> upParticleFindingTask = nullptr);
    std::shared_ptr<AsyncParticleFindingTask> m_spParticleFindingTask;
    std::future<void> m_fuParticleFindingTask;
    std::vector<FoundParticle>  launchTask (std::shared_ptr<AsyncParticleFindingTask> spParticleFindingTask);
    void cancelTask();
};

// Helper functions
cv::cuda::GpuMat GetContinuousGpuMat( cv::Mat& m );
void RemapImage( cv::Mat& img, double dMin, double dMax );            // Rescale image values
void RemapImage( cv::cuda::GpuMat& img, double dMin, double dMax );    // Rescale image values
void ShowImage( cv::Mat& img );                                        // Show a host image
void ShowImage( cv::cuda::GpuMat& img );                            // Show a GPU image

#include <cuda_runtime.h>
class CudaStopWatch
{
    std::string name;
    cudaEvent_t start, stop;
public:
    CudaStopWatch (std::string n) :
        name (n)
    {
        cudaEventCreate (&start);
        cudaEventCreate (&stop);
        cudaEventRecord (start);
    }

    ~CudaStopWatch ()
    {
        cudaEventRecord (stop);
        cudaEventSynchronize (stop);

        // Print out the elapsed time
        float mS (0.f);
        cudaEventElapsedTime (&mS, start, stop);
        printf ("%s took %f mS to execute\n", name.c_str (), mS);
    }
};