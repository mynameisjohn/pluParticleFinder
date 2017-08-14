#include "ParticleFinder.h"

#include <FreeImage.h>
#include <algorithm>
#include <set>
#include <fstream>

#include <functional>

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
	typedef std::function<_Res( _ArgTypes... )> function_type;

	static void bind( function_type&& f )
	{
		instance().fn_.swap( f );
	}

	static void bind( const function_type& f )
	{
		instance().fn_ = f;
	}

	static _Res invoke( _ArgTypes... args )
	{
		return instance().fn_( args... );
	}

	typedef decltype( &fun_ptr_helper::invoke ) pointer_type;
	static pointer_type ptr()
	{
		return &invoke;
	}

private:
	static fun_ptr_helper& instance()
	{
		static fun_ptr_helper inst_;
		return inst_;
	}

	fun_ptr_helper() {}

	function_type fn_;
};

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr( const std::function<_Res( _ArgTypes... )>& f )
{
	fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind( f );
	return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function( T *t )
{
	return { t };
}

// These are the default testbed params
ParticleFinder::ParticleFinder() :
	m_nGaussFiltRadius( 6 ),
	m_nDilationRadius( 3 ),
	m_fHWHM( 4 ),
	m_fParticleThreshold( 0.005f )
{}

ParticleFinder::~ParticleFinder() 
{
	cancelTask();
}

int ParticleFinder::GetNumImages() const 
{
	return (int)m_vdInputImages.size();
}

bool ParticleFinder::Initialize(std::list<std::string> liStackPaths, int nStartOfStack, int nEndOfStack, bool bDoUserInput /*= false*/)
{
	// For each tiff stack
	int nSlices(0);
	std::cout << "Starting to load slices..." << std::endl;

	// Load the images - this could be done
	// concurrently to the CenterFind DSP
	for (const std::string& strStackPath : liStackPaths)
	{
		// Attempt to open multibitmap
		FIMULTIBITMAP * FI_Input = FreeImage_OpenMultiBitmap(FIF_TIFF, strStackPath.c_str(), 0, 1, 1, TIFF_DEFAULT);
		if (FI_Input == nullptr)
		{
			std::cout << "Error loading stack " << strStackPath << std::endl;
			continue;
		}
		std::cout << "Loading stack " << strStackPath << std::endl;

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
				if (m_vdInputImages.empty())
				{
					// Initialize all other mats to zero on device (may be superfluous)
					m_dFilteredImg = GpuMat(imgSize, CV_32F, 0.f);
					m_dDilatedImg = GpuMat(imgSize, CV_32F, 0.f);
					m_dTmpImg = GpuMat(imgSize, CV_32F, 0.f);
					m_dLocalMaxImg = GpuMat(imgSize, CV_32F, 0.f);
					m_dTmpImg = GpuMat(imgSize, CV_32F, 0.f);

					// It's in our best interest to ensure these are continuous
					cv::cuda::createContinuous(imgSize, CV_32F, m_dThreshImg);
					cv::cuda::createContinuous(imgSize, CV_8U, m_dParticleImg);
					m_dThreshImg.setTo(0);
					m_dParticleImg.setTo(0);
				}

				// Convert FIBITMAP to 24 bit RGB, store inside cv::Mat's buffer
				FreeImage_ConvertToRawBits(m.data, image, m.step, 24, 0xFF, 0xFF, 0xFF, true);

				// Upload to device (should this get a separate buffer?)
				GpuMat d_InputImg;
				d_InputImg.upload(m);

				// Convert to greyscale float, store in our input buffer
				cv::cuda::cvtColor(d_InputImg, m_dTmpImg, CV_RGB2GRAY);
				m_dTmpImg.convertTo(d_InputImg, CV_32F, 1. / 0xFF);
				m_vdInputImages.push_back(d_InputImg);

				// Inc slice count
				nSlices++;
			}
			else
			{
				std::cout << "Error loading slice " << ixSlice << " of stack " << strStackPath << std::endl;
			}

			// Print something out every 10 slices
			if (ixSlice % 10 == 0)
				std::cout << "On image " << ixSlice << " of stack " << strStackPath << "..." << std::endl;
		}

		// Close multibitmap
		FreeImage_CloseMultiBitmap(FI_Input);
	}

	if (!m_vdInputImages.empty()) 
	{
		m_Solver.Reset();
		return true;
	}

	return false;
}

void ParticleFinder::cancelTask() 
{
	if (m_spParticleFindingTask)
	{
		{
			LockMutex(m_spParticleFindingTask->muData);
			m_spParticleFindingTask->bCancel = true;
		}

		// Wait till it's done
		while (true)
		{
			LockMutex(m_spParticleFindingTask->muData);
			if (m_spParticleFindingTask->bIsDone)
				break;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		m_spParticleFindingTask = nullptr;
	}
}

std::vector<ParticleFinder::FoundParticle> ParticleFinder::Execute(std::shared_ptr<AsyncParticleFindingTask> spParticleFindingTask /*= nullptr*/)
{
	// Construct filters now
	// make sure that we have valid params)
	if (m_vdInputImages.empty() || m_nDilationRadius == 0 || m_nGaussFiltRadius == 0)
		return{};

	// If we have a task going, cancel it now
	if (m_spParticleFindingTask)
	{
		cancelTask();
	}

	// If an async task was passed in, we return empty
	// and kick the task off with the supplied params
	if (spParticleFindingTask)
	{
		// Launch thread
		m_spParticleFindingTask = spParticleFindingTask;
		std::thread([this, spParticleFindingTask]()
		{
			// Set DSP params
			SetGaussianRadius(m_spParticleFindingTask->nGaussianRadius);
			SetFWHM(m_spParticleFindingTask->fHWHM);
			SetDilationRadius(m_spParticleFindingTask->nDilationRadius);
			SetParticleThreshold(m_spParticleFindingTask->fParticleThreshold);
			GetSolver()->SetMaskRadius(m_spParticleFindingTask->nMaskRadius);
			GetSolver()->SetFeatureRadius(m_spParticleFindingTask->nFeatureRadius);
			GetSolver()->SetMinSliceCount(m_spParticleFindingTask->nMinSliceCount);
			GetSolver()->SetMaxSliceCount(m_spParticleFindingTask->nMaxSliceCount);
			GetSolver()->SetNeighborRadius(m_spParticleFindingTask->nNeighborRadius);
			GetSolver()->SetMaxLevel(m_spParticleFindingTask->nMaxLevel);
			GetSolver()->Init();
			
			// Make sure this is empty
			m_spParticleFindingTask->mapFoundSliceToParticles.clear();

			for ( size_t i = 0; i < m_vdInputImages.size(); i++)
			{
				{	// Check to see if we're cancelled
					// TODO clear solver's found particles
					LockMutex(m_spParticleFindingTask->muData);
					if (m_spParticleFindingTask->bCancel)
					{
						GetSolver()->Reset();
						m_spParticleFindingTask->bIsDone = true;
						return;
					}
				}

				// Find particles in image
				doDSPAndFindParticlesInImg((int)i, m_vdInputImages[i], true);

				// Update task with found particles (we have to look at an earlier slice,
				// as the particle center won't be in the slice we just processed)
				if (i >= m_spParticleFindingTask->nMaxSliceCount)
				{
					int ixFindParticles = (int)(i - m_spParticleFindingTask->nMaxSliceCount);
					{
						LockMutex(m_spParticleFindingTask->muData);
						m_spParticleFindingTask->nLastImageProcessed = ixFindParticles;
						m_spParticleFindingTask->mapFoundSliceToParticles[ixFindParticles] = m_Solver.GetFoundParticlesInSlice( ixFindParticles );
					}
				}
			}

			{	// Verdun
				std::lock_guard<std::mutex> lg(m_spParticleFindingTask->muData);
				m_spParticleFindingTask->bIsDone = true;
			}

			return;
		}
		).detach();

		return{};
	}
	else
	{
		GetSolver()->Init();

		// Otherwise we do the DSP with our current params and return found particles
		for (size_t i = 0; i < m_vdInputImages.size(); i++)
		{
			doDSPAndFindParticlesInImg((int)i, m_vdInputImages[i], true);
		}

		// Return all found particles
		return m_Solver.GetFoundParticles();
	}
}

void ParticleFinder::getUserInput( GpuMat d_Input )
{
	// Window name
	std::string windowName = "Particle Finder Parameters";

	// Trackbar Names
	std::string gaussRadiusTBName = "Gaussian Radius";
	std::string hwhmTBName = "Half-Width at Half-Maximum ";
	std::string dilationRadiusTBName = "Dilation Radius";
	std::string particleThreshTBName = "Particle Intensity Threshold";

	// We need pointers to these ints
	std::map<std::string, int> mapParamValues = {
		{ gaussRadiusTBName, 6 },	// These are the
		{ hwhmTBName, 4 },			// default values
		{ dilationRadiusTBName, 3 },// specified in the
		{ particleThreshTBName, 5 } // PLuTARC_testbed
	};
	
	const float trackBarResolution = 1000;
	for ( auto& it : mapParamValues )
		it.second *= trackBarResolution;
	
	try
	{
		// Trackbar callback, implemented below
		std::function<void( int, void * )> trackBarCallback = [&]( int pos, void * priv )
		{
			// Assign DSP params
			m_nGaussFiltRadius = mapParamValues[gaussRadiusTBName] / trackBarResolution;
			m_fHWHM = mapParamValues[hwhmTBName] / trackBarResolution;
			m_nDilationRadius = mapParamValues[dilationRadiusTBName] / trackBarResolution;
			m_fParticleThreshold = mapParamValues[particleThreshTBName] / trackBarResolution;

			// TODO prevent bad values from existing

			GpuMat d_InputCirc = d_Input.clone();

			// do DSP, get particles
			std::vector<FoundParticle> vParticlesInImg;
			doDSPAndFindParticlesInImg( 0, d_Input, false, &vParticlesInImg, true );

			// Draw circles in local max image
			if ( !vParticlesInImg.empty() )
			{
				cv::Mat hImg;
				d_InputCirc.download( hImg );
				// This offset was needed for some reason
				int dX( -3 ), dY( -3 );
				for ( FoundParticle& fp : vParticlesInImg)
				{
					cv::circle( hImg, cv::Point( fp.fPosX + dX, fp.fPosY + dY ), 5, cv::Scalar( 0 ), -1 );
				}
				d_InputCirc.upload( hImg );
			}

			// Show several of our images
			// returns formatted images for display
			auto makeDisplayImage = []( GpuMat& in )
			{
				GpuMat out;
				in.convertTo( out, CV_32F );
				RemapImage( out, 0, 1 );
				return out;
			};

			// Create larger display image (4 images, corner to corner)
			cv::Size dataSize = d_Input.size();
			cv::Size dispSize = dataSize;
			dispSize *= 2;	// Multiply by two in x and y
			GpuMat displayMat( dispSize, CV_32F, 0.f );

			// Display regions
			cv::Rect topLeft( { 0, 0 }, dataSize );
			cv::Rect topRight( cv::Rect( { dataSize.width, 0 }, dataSize ) );
			cv::Rect bottomLeft( { 0, dataSize.height }, dataSize );
			cv::Rect bottomRight( { dataSize.width, dataSize.height }, dataSize );

			// Copy all images to display image in correct place
			d_Input.copyTo( displayMat( topLeft ) );
			makeDisplayImage( m_dFilteredImg ).copyTo( displayMat( topRight ) );
			makeDisplayImage( m_dDilatedImg ).copyTo( displayMat( bottomLeft ) );
			makeDisplayImage( d_InputCirc ).copyTo( displayMat( bottomRight ) );

			// Show new image
			cv::resizeWindow( windowName, dispSize.width, dispSize.height );
			cv::imshow( windowName, displayMat );
		};

		// Create window, just show input first
		cv::namedWindow( windowName, cv::WINDOW_OPENGL );

		// Create trackbars
		auto createTrackBar = [&mapParamValues, windowName, &trackBarCallback]( std::string tbName, int maxVal )
		{
			auto it = mapParamValues.find( tbName );
			if ( it != mapParamValues.end() )
			{
				cv::createTrackbar( tbName, windowName, &mapParamValues[tbName], maxVal, get_fn_ptr<0>( trackBarCallback ) );
			}
		};

		createTrackBar( gaussRadiusTBName, 15 * trackBarResolution );
		createTrackBar( hwhmTBName, 15 * trackBarResolution );
		createTrackBar( dilationRadiusTBName, 15 * trackBarResolution );
		createTrackBar( particleThreshTBName, 15 * trackBarResolution );

		// Call the callback on our own, just to pump things and show the images
		trackBarCallback( 0, nullptr );

		// Wait while user sets things until they press a key (any key?)
		cv::waitKey();

		// Destroy window
		cv::destroyWindow( windowName );
	}
	catch ( cv::Exception e )
	{
		std::cout << e.what() << std::endl;
		std::cout << "Error creating user interface! Using default parameters\n" << std::endl;
	}
}

int ParticleFinder::doDSPAndFindParticlesInImg(int ixSlice, GpuMat d_Input, bool bLinkParticles /*= true*/, std::vector<FoundParticle> * pFoundParticles /*= nullptr*/, bool bResetKernels /*= false*/)
{
	if ( bResetKernels || m_dCircleFilter.empty() || m_dDilationKernel.empty() )
	{	
		std::cout << "Constructing DSP kernels" << std::endl;

		// Create circle image
		int nBPDiameter = 2 * m_nGaussFiltRadius + 1;
		cv::Mat h_Circle = cv::Mat::zeros( cv::Size( nBPDiameter, nBPDiameter ), CV_32F );
		cv::circle( h_Circle, cv::Size( m_nGaussFiltRadius, m_nGaussFiltRadius ), m_nGaussFiltRadius, 1.f, -1 );
		m_dCircleFilter = cv::cuda::createLinearFilter( CV_32F, CV_32F, h_Circle );

		// Create Gaussian Filter and normalization scale
		const cv::Size bpFilterSize( nBPDiameter, nBPDiameter );
		const double dSigma = (double) m_fHWHM / ( ( sqrt( 2 * log( 2 ) ) ) );
		m_dGaussFilter = cv::cuda::createGaussianFilter( CV_32F, CV_32F, bpFilterSize, dSigma );
		double dGaussianScale = 1 / ( 3 * pow( m_nGaussFiltRadius, 2 ) );

		// Create dilation mask
		int nDilationDiameter = 2 * m_nDilationRadius + 1;
		cv::Mat h_Dilation = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( nDilationDiameter, nDilationDiameter ) );

		// Create dilation kernel from host kernel (only single byte supported? why nVidia why)
		m_dDilationKernel = cv::cuda::createMorphologyFilter( cv::MORPH_DILATE, CV_32F, h_Dilation );
	}

	std::cout << "Finding and linking particles in slice " << ixSlice << std::endl;

	double dGaussianScale = 1 / ( 3 * pow( m_nGaussFiltRadius, 2 ) );

	// Apply low/high pass filters (gaussian and circle kernels, resp.)
	m_dGaussFilter->apply( d_Input, m_dFilteredImg );
	//ShowImage( m_dFilteredImg );

	m_dCircleFilter->apply( d_Input, m_dTmpImg );
	m_dTmpImg.convertTo( m_dTmpImg, CV_32F, dGaussianScale );
	//ShowImage( m_dTmpImg );

	// subtract tmp from bandpass to get filtered output
	cv::cuda::subtract( m_dFilteredImg, m_dTmpImg, m_dFilteredImg );
	//ShowImage( m_dFilteredImg );

	// Any negative values become 0
	cv::cuda::threshold( m_dFilteredImg, m_dFilteredImg, 0, 1, cv::THRESH_TOZERO );
	//ShowImage( m_dFilteredImg );

	// Remap this from 0 to 100, so we have a normalized
	// range of particle intenstiy thresholds
	RemapImage( m_dFilteredImg, 0, 100 );

	// Reset thresh to base value
	m_dThreshImg.setTo( cv::Scalar( m_fParticleThreshold ) );

	// Store max of (m_fParticleThreshold, m_dFilteredImg) in m_dThreshImg
	cv::cuda::max( m_dThreshImg, m_dFilteredImg, m_dThreshImg );
	//ShowImage( m_dThreshImg );

	// Initialize dilated image to threshold value
	m_dDilatedImg.setTo( cv::Scalar( m_fParticleThreshold ) );

	// Apply dilation
	m_dDilationKernel->apply( m_dThreshImg, m_dDilatedImg );
	//( m_dDilatedImg );

	// Subtract filtered image from dilated (negates pixels that are not local maxima)
	cv::cuda::subtract( m_dFilteredImg, m_dDilatedImg, m_dLocalMaxImg );

	// Exponentiate m_dLocalMaxImg (sends negative values to low numbers, lm to 0)
	cv::cuda::exp( m_dLocalMaxImg, m_dLocalMaxImg );

	// Threshold exponentiated image - we are left with only local maxima as nonzero pixels
	cv::cuda::threshold( m_dLocalMaxImg, m_dLocalMaxImg, 1 - 0.0001f, 1, cv::THRESH_BINARY );
	//ShowImage( m_dLocalMaxImg );

	// Cast to uchar, store as particle image (values are still 0 or 1, so no scale needed)
	m_dLocalMaxImg.convertTo( m_dParticleImg, CV_8U );

	return m_Solver.FindParticlesInImage( ixSlice, d_Input, m_dFilteredImg, m_dThreshImg, m_dParticleImg, bLinkParticles, pFoundParticles );
}

void ParticleFinder::SetGaussianRadius( int nGaussianRadius )
{
	m_nGaussFiltRadius = nGaussianRadius;
}

void ParticleFinder::SetDilationRadius( int nDilationRadius )
{
	m_nDilationRadius = nDilationRadius;
}

void ParticleFinder::SetFWHM( float fHWHM )
{
	m_fHWHM = fHWHM;
}

void ParticleFinder::SetParticleThreshold( float fParticelThreshold )
{
	m_fParticleThreshold = fParticelThreshold;
}

int ParticleFinder::GetGaussianRadius() const
{
	return m_nGaussFiltRadius;
}

int ParticleFinder::GetDilationRadius() const
{
	return m_nDilationRadius;
}

float ParticleFinder::GetFWHM() const
{
	return m_fHWHM;
}

float ParticleFinder::GetParticleThreshold() const
{
	return m_fParticleThreshold;
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
	float b = m1 - a * m0;
	img.convertTo( img, img.type(), a, b );
}

void RemapImage( cv::Mat& img, double m1, double M1 )
{
	double m0( 1 ), M0( 2 );
	cv::minMaxLoc( img, &m0, &M0 );
	double a = ( M1 - m1 ) / ( M0 - m0 );
	float b = m1 - a * m0;
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