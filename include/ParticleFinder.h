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
// to the Solver member
class ParticleFinder
{
	///////////////////////////////////////////////////////////////////
	// Useful typedef
	using GpuMat = cv::cuda::GpuMat;
	std::vector<GpuMat> m_vdInputImages;	// Buffer used for input

	///////////////////////////////////////////////////////////////////
	// DSP parameters
	int m_nGaussFiltRadius;		// Radius of Gaussian (low-pass) Filter
	int m_nDilationRadius;		// Radius of dilation operation
	float m_fHWHM;				// Half-Width at Half Maximum (describes gaussian kernel)
	float m_fParticleThreshold;	// Threshold intensity for particles in image
	
	// Solver class - looks at particle in successive images and
	//	1) determines particle locations in the current slices 
	//	2) uses the locations to track particles across multiple slices
	//	3) determines particle "centers" (located at slice with max intensity)
	//	4) discards invalid particles
	// Implemented in .cu/cuh files because it runs some thrust/CUDA code
public:
	struct FoundParticle
	{
		float fPosX;
		float fPosY;
		float fPosZ;
		float fIntensity;
	};

private:
	class Solver
	{
		struct impl;
		std::unique_ptr<Solver::impl> m_upSolverImpl;
	public:
		// Initializes DSP params to default testbed params
		Solver();

		// Destructor must be implemented... for impl
		~Solver();

		// Find particles in a given image, return the count of particles found
		int FindParticlesInImage(int nSliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg,
			bool bLinkParticles = true, std::vector<FoundParticle> * pParticlesInImg = nullptr );

		// Get all found particles
		std::vector<FoundParticle> GetFoundParticles() const;

		std::vector<FoundParticle> GetFoundParticlesInSlice(int ixSlice) const;

		void Init();

		// Reset current count of found particles
		void Reset();

		// Getters and setters for the solver params (implementd by solver::impl)
		int GetMaskRadius() const;		// The radius of our particle mask kernels
		int GetFeatureRadius() const;	// The radius within the image we'd like to consier
		int GetMinSliceCount() const;	// The minimum # of slices we require to contribute to a particle
		int GetMaxSliceCount() const;	// The maximum # of slices we allow to contribute to a particle
		int GetNeighborRadius() const;	// The radius in which we search for new particles
		int GetMaxLevel() const;		// The subdivision level we use to spatially partition previous particles
		void SetMaskRadius( int mR );
		void SetFeatureRadius( int fR );
		void SetMinSliceCount( int nMinSC );
		void SetMaxSliceCount( int nMaxSC );
		void SetNeighborRadius( int nR );
		void SetMaxLevel( int mL );
	} m_Solver;

	// Internal buffers - these are of the same
	// dimensions, but may be different data types
	GpuMat m_dFilteredImg;	// Post Gaussian Bandpass
	GpuMat m_dDilatedImg;	// Post dilation
	GpuMat m_dLocalMaxImg;	// Local Max image
	GpuMat m_dParticleImg;	// Particle Image
	GpuMat m_dThreshImg;	// Thresholded Particle image
	GpuMat m_dTmpImg;		// Temporary image

	cv::Ptr<cv::cuda::Filter> m_dGaussFilter;		// Gaussian Filter
	cv::Ptr<cv::cuda::Filter> m_dCircleFilter;		// Circle Filter
	cv::Ptr<cv::cuda::Filter> m_dDilationKernel;	// Dilation Filter
	
	// This actual detects particles in the input image
	// the result is a contiguous black and white
	// image that can be used in particle detection
	int doDSPAndFindParticlesInImg( int ixSlice, GpuMat d_Input, bool bLinkParticles = true, std::vector<FoundParticle> * pFoundParticles = nullptr, bool bResetKernels = false );

	void getUserInput( GpuMat d_Input );
	
public:
	// Initializes DSP params to default testbed params
	ParticleFinder();
	~ParticleFinder();

	// Simple getter for solver to make it accessible
	Solver * GetSolver() { return &m_Solver; }

	// The other way is to use these setters
	void SetGaussianRadius( int nGaussianRadius );
	void SetDilationRadius( int nDilation );
	void SetFWHM( float fHWHM );
	void SetParticleThreshold( float fParticleThreshold );
	int GetGaussianRadius() const;
	int GetDilationRadius() const;
	float GetFWHM() const;
	float GetParticleThreshold() const;

	// Runs the DSP and particle solving / linking
	// code on whatever TIFF stacks have been loaded
	// using whatever DSP params have been set
	// Returns the # of particles found acrosds all stacks


	bool Initialize(std::list<std::string> liStackPaths, int nStartOfStack, int nEndOfStack, bool bDoUserInput = false);

	int GetNumImages() const;

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
		int nMaxLevel;
		std::map<int, std::vector<FoundParticle>> mapFoundSliceToParticles;
	};
	std::vector<FoundParticle> Execute(std::shared_ptr<AsyncParticleFindingTask> upParticleFindingTask = nullptr);
	std::shared_ptr<AsyncParticleFindingTask> m_spParticleFindingTask;
	std::future<void> m_fuParticleFindingTask;
	void cancelTask();
};

// Helper functions
cv::cuda::GpuMat GetContinuousGpuMat( cv::Mat& m );
void RemapImage( cv::Mat& img, double dMin, double dMax );			// Rescale image values
void RemapImage( cv::cuda::GpuMat& img, double dMin, double dMax );	// Rescale image values
void ShowImage( cv::Mat& img );										// Show a host image
void ShowImage( cv::cuda::GpuMat& img );							// Show a GPU image