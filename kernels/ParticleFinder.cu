#include "ParticleFinder.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/count.h>

#define SOLVER_DEVICE 1

// Grid cell, used as a 2-D spatial data 
// structure to help searching for particle matches
// The members are the first and last particle index
// within the buffer the particles live in
struct Cell
{
    int lower;
    int upper;
};

// Our device particle struct
struct Particle
{
    // Intensity state, tracked across
    // multiple slices to determine where particle ends
    enum class State
    {
        NO_MATCH = 0,
        INCREASING,
        DECREASING,
        SEVER
    };
    float x;                        // X pos
    float y;                        // Y pos
    float z;                        // Z pos
    float i;                        // Intensity
    float peakIntensity;            // Peak intensity (center)
    int ixCenterSlice;                // Slice index of peak intensity (center)
    int ixLastSlice;                // Index of last contributing slice
    int nContributingSlices;        // # of slices contributing
    State pState;                    // Current intensity state

    __host__ __device__
        Particle( float x = -1.f, float y = -1.f, float i = -1.f, int idx = -1 ) :
        x( x ),
        y( y ),
        z( (float) idx ),
        i( i ),
        peakIntensity( i ),
        ixCenterSlice( 0 ),
        ixLastSlice( idx ),
        nContributingSlices( 1 ),
        pState( State::NO_MATCH )
    {}
};

// Helper for making thrust zip iterators
template <typename ... Args>
auto MakeZipIt( const Args&... args ) -> decltype( thrust::make_zip_iterator( thrust::make_tuple( args... ) ) )
{
    return thrust::make_zip_iterator( thrust::make_tuple( args... ) );
}

// Solver implementation
struct ParticleFinder::Solver::impl
{
    // Useful typedefs of mine
#if SOLVER_DEVICE
    using UcharVec = thrust::device_vector < unsigned char >;
    using UcharPtr = thrust::device_ptr < unsigned char >;
    using IntVec = thrust::device_vector < int >;
    using IntPtr = thrust::device_ptr < int >;
    using FloatVec = thrust::device_vector < float >;
    using Floatptr = thrust::device_ptr < float >;
    using ParticleVec = thrust::device_vector < Particle >;
    using ParticlePtrVec = thrust::device_vector < Particle * >;
    using FoundParticleVec = thrust::device_vector <ParticleFinder::FoundParticle>;
    using Img = cv::cuda::GpuMat;
#else
    // I use these for host debugging
    using UcharVec = thrust::host_vector < unsigned char >;
    using UcharPtr = unsigned char *;
    using IntVec = thrust::host_vector < int >;
    using IntPtr = int *;
    using FloatVec = thrust::host_vector < float >;
    using Floatptr = float *;
    using ParticleVec = thrust::host_vector < Particle >;
    using ParticlePtrVec = thrust::host_vector < Particle * >;
    using FoundParticleVec = thrust::host_vector <ParticleFinder::FoundParticle>;
    using Img = cv::Mat;
#endif

    impl();
    void Init();
    int FindParticlesInImage(int nSliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, bool bLinkParticles, std::vector<FoundParticle> * pParticlesInImg);
    std::vector<FoundParticle> GetFoundParticles() const;
    std::vector<FoundParticle> GetFoundParticlesInSlice(int ixSlice) const;

    int m_uMaskRadius;                    // The radius of our particle mask kernels
    int m_uFeatureRadius;                // The radius within the image we'd like to consier
    int m_uMinSliceCount;                // The minimum # of slices we require to contribute to a particle
    int m_uMaxSliceCount;                // The maximum # of slices we allow to contribute to a particle
    int m_uNeighborRadius;                // The radius in which we search for new particles
    int m_uMaxLevel;                    // The subdivision level we use to spatially partition previous particles

    Img m_dCircleMask;                    // The circle mask, just a circle of radius m_uMaskRadius, each value is 1
    Img m_dRadXKernel;                    // The x mask, used to calculate an offset to the x coordinate
    Img m_dRadYKernel;                    // The y mask, used to calculate an offset to the y coordinate
    Img m_dRadSqKernel;                    // The r2 mask, used to calculate some value that I don't really understand

    size_t m_uCurPrevParticleCount;        // The current tally of previous particles to search through
    ParticleVec m_dPrevParticleVec;        // The vector of previously found particles

    IntVec m_dGridCellLowerBoundVec;    // The lower bound vector of particles to search
    IntVec m_dGridCellUpperBoundVec;    // The upper bound vector of particles to search

    // These are private functions that actually do the solving
    void createGridCells( int N );
    void updateMatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec, int sliceIdx );
    void mergeUnmatchedParticles( ParticleVec& d_UnmatchedParticleVec, int N );
    size_t cullExistingParticles( int curSliceIdx );        // Remove found particles if they are deemed to be noise
    ParticleVec findNewParticles( UcharVec& d_ParticleImgVec, Floatptr pThreshImg, int N, int sliceIdx );
    ParticlePtrVec findParticleMatches( ParticleVec& d_NewParticleVec, int N, int sliceIdx );
    ParticleVec consolidateUnmatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec );
};

// IsParticleInState predicate
// Construct
struct IsParticleInState
{
    Particle::State PS;
    IsParticleInState( Particle::State ps) :
        PS( ps )
    {}
    __host__ __device__
    bool operator()( const Particle p ){
        return p.pState == PS;
    }
};

// Converts the ParticleClass (which has device constructors)
// to the host only FoundParticle class
struct Particle2FoundParticle
{
    __host__ __device__
    ParticleFinder::FoundParticle operator()( const Particle& p )
    {
        ParticleFinder::FoundParticle ret;
        ret.fIntensity = 654;
        ret.fPosZ = 12;
        ret.fPosX= p.x;
        ret.fPosY = p.y;
        return ret;
    }
};
struct Particle2Float4
{
    __host__ __device__
    float4 operator()( const Particle& p )
    {
        return float4{ p.x, p.y, p.z, p.i };
    }
};

// Turn an x,y coord into a grid index given the image dimension N
// and maximum subdivisions of that image m (N is a power of 2)
__host__ __device__
int pixelToGridIdx( float x, float y, int N, int m )
{
    const int cellSize = N >> m;
    const int cellCount = N / cellSize;

    int cellX = x / cellSize;
    int cellY = y / cellSize;

    int cellIdx = cellX + cellCount * cellY;
    return cellIdx;
}

// Invoke the above given a particle
__host__ __device__
int pixelToGridIdx( Particle p, int N, int m )
{
    return pixelToGridIdx( p.x, p.y, N, m );
}

// Our particle to grid index operator
struct PixelToGridIdx : public thrust::unary_function < Particle, int >
{
    int N; // Image size
    int M; // division level

    PixelToGridIdx( int n, int m ) :N( n ), M( m ) {}

    __host__ __device__
        int operator()( Particle p )
    {
        // Just call the function
        return pixelToGridIdx( p, N, M );
    }
};

// Particle removal predicate
struct MaybeRemoveParticle
{
    int sliceIdx;    // Current slice index
    int minSlices;    // The minimum # of slices we require to contribute to a particle
    MaybeRemoveParticle( int s, int m ) : 
        sliceIdx( s ), minSlices( m ) 
    {}

    __host__ __device__
        bool operator()( const Particle p )
    {
        // If the particle is 2 (should be var) slices away from us, and it isn't in a sever state or is but has too few particles, return true
        return ( sliceIdx - p.ixLastSlice > 2 && ( p.pState != Particle::State::SEVER || p.nContributingSlices < minSlices ) );
    }
};

// Particle detection predicate
struct IsParticleAtIdx
{
    int N;            // Image dimension
    int featureRad;    // Image feature radius

    IsParticleAtIdx( int n, int k ) : N( n ), featureRad( k ) {}

    template <typename tuple_t>
    __host__ __device__
        bool operator()( tuple_t T )
    {
        // Unpack tuple
        unsigned char val = thrust::get<0>( T );
        int idx = thrust::get<1>( T );

        // get xy coords from pixel index
        int x = idx % N;
        int y = idx / N;

        // We care if the pixel is nonzero and its within the kernel radius
        return ( val != 0 ) && ( x > featureRad ) && ( y > featureRad ) && ( x + featureRad < N ) && ( y + featureRad < N );
    }
};

// Simple predicate that detects whether a pointer in this tuple is null
struct CheckIfMatchIsNull
{
    template <typename tuple_t>
    __host__ __device__
        bool operator()( const tuple_t T )
    {
        Particle * pMatch = thrust::get<1>( T );
        return pMatch == nullptr;
    }
};

// Opposite of above
struct CheckIfMatchIsNotNull
{
    template <typename tuple_t>
    __host__ __device__
        bool operator()( const tuple_t T )
    {
        Particle * pMatch = thrust::get<1>( T );
        return pMatch != nullptr;
    }
};

// Turn an index at which we've detected a particle (above) into a real particle
struct MakeParticleFromIdx
{
    int sliceIdx;        // Current slice index
    int kernelRad;        // Kernel (mask) radius
    int N;                // Image dimension
    float * lmImg;        // pointer to image in which particle was detected
    float * circMask;    // pointer to circle mask
    float * rxMask;        // pointer to x offset mask
    float * ryMask;        // pointer to y offset mask
    float * rSqMask;    // pointer to r2 mask 
    MakeParticleFromIdx( int sIdx, int n, int kRad, float * lm, float * cK, float * xK, float * yK, float * sqK ) :
        sliceIdx( sIdx ),
        N( n ),
        kernelRad( kRad ),
        lmImg( lm ),
        circMask( cK ),
        rxMask( xK ),
        ryMask( yK ),
        rSqMask( sqK )
    {}

    __host__ __device__
        Particle operator()( int idx )
    {
        // Grab x, y values
        int x = idx % N;
        int y = idx / N;

        // Make tmp pointers to our masks and advance them
        // as we iterate to perform the multiplication
        float * tmpCircPtr = circMask;
        float * tmpXPtr = rxMask;
        float * tmpYPtr = ryMask;

        // To be calculated
        float total_mass( 0 );
        float x_offset( 0 ), y_offset( 0 );

        // Apply the mask as a multiplcation
        for ( int iY = -kernelRad; iY <= kernelRad; iY++ )
        {
            // For y, go down then up
            float * ptrY = &lmImg[idx + ( N * iY )];
            for ( int iX = -kernelRad; iX <= kernelRad; iX++ )
            {
                // Get the local max img value
                float lmImgVal = ptrY[iX];

                // Multiply by mask value, sum, advance mask pointer
                total_mass += lmImgVal * ( *tmpCircPtr++ );
                x_offset += lmImgVal * ( *tmpXPtr++ );
                y_offset += lmImgVal * ( *tmpYPtr++ );
            }
        }

        // Calculate x val, y val
        float x_val = float( x ) + x_offset / total_mass;
        float y_val = float( y ) + y_offset / total_mass;

        // Construct particle and return
        Particle p( x_val, y_val, total_mass, sliceIdx );
        return p;
    }
};

// This operator searches through the grid cells of our previous particle and tries to find a match
// We use these grid cells as an optimization - rather than search all particles in the previous stack,
// search only particles in our cell and in the nine cells 
struct ParticleMatcher
{
    int N;                        // Image dimension
    int M;                        // Maximum # of subdivisions
    int sliceIdx;                // current slice index
    int maxStackCount;            // max # of slices that can contribute to a particle
    int cellSize;                // The # of pixels per cell (1D)
    int cellDim;                // The # of cells per image (1D)
    float neighborRadius;        // radius around which we search for matches
    int * cellLowerBound;        // Pointer to lower bound of prev particle range (sorted by index)
    int * cellUpperBound;        // pointer to upper bound of prev particle range (sorted by index)
    Particle* prevParticles;    // Pointer to the vector of previous particles
    ParticleMatcher( int n, int m, int s, int mSC, int cC, int nR, int * cLB, int * cUB, Particle * pP ) :
        N( n ),
        M( m ),
        sliceIdx( s ),
        maxStackCount( mSC ),
        neighborRadius( nR ),
        cellLowerBound( cLB ),
        cellUpperBound( cUB ),
        prevParticles( pP )
    {
        cellSize = N >> M;       // # of pixels per cell dim
        cellDim = N / cellSize;  // # of cells per image dim
    }

    // Returns null if no match is found
    __host__ __device__
        Particle * operator()( Particle newParticle )
    {
        // The values in this array are grid indices, and we always search the center
        // The other values will be left negative until we determine the y should be searched (below)
        // There are a total of 9 cells we may have to search - last is sentinel
        int cellIndices[10] = { pixelToGridIdx( newParticle, N, M ), -1, -1, -1, -1, -1, -1, -1, -1, -1 };

        // See if we need to search neighbors
        int neighborIdx = 1;                    // Index in cellIndices
        int cellX = cellIndices[0] % cellDim; // X position of center cell
        int cellY = cellIndices[0] / cellDim; // Y position of center cell

        // If we aren't on the left edge of the image
        if ( cellX != 0 )
        {
            // And we are far enough away from the left grid border
            int leftGridBorder = cellX * cellSize;
            if ( (int) newParticle.x - neighborRadius < leftGridBorder )
                cellIndices[neighborIdx++] = cellIndices[0] - 1;
        }

        // Similar for other directions
        if ( cellX != cellDim - 1 )
        {
            float rightGridBorder = ( cellX + 1 ) * cellSize;
            if ( (int) newParticle.x + float( neighborRadius ) > rightGridBorder )
                cellIndices[neighborIdx++] = cellIndices[0] + 1;
        }

        // "Bottom" is actually the top, when you think about it... right?
        // Make sure you haven't messed that up, please
        // The neigbor cell we're looking for is the center offset by the
        // cell size, which is the 1d size of a cell in pixels
        if ( cellY != 0 )
        {
            float bottomGridBorder = cellY * cellSize;
            if ( newParticle.y - neighborRadius < bottomGridBorder )
                cellIndices[neighborIdx++] = cellIndices[0] - cellDim;
        }

        if ( cellY != cellDim - 1 )
        {
            float top = ( cellY + 1 ) * cellSize;
            if ( newParticle.y + neighborRadius > top )
                cellIndices[neighborIdx++] = cellIndices[0] + cellDim;
        }

        // Why arent you checking the diagonal corners? You hack. 

        // For every cell we've decided to search (break when index is negative)
        const Particle * pBestMatch = nullptr;
        for ( int c = 0; cellIndices[c] >= 0; c++ )
        {
            // See if every particle in that cell is a match
            // It would be nice to parallelize around this, 
            // but whether or not that's worth it depends on how populated the grid cells are
            int cellIdx = cellIndices[c];
            int lower = cellLowerBound[cellIdx];    // First particle to search
            int upper = cellUpperBound[cellIdx];    // End of range to search
            for ( int p = lower; p < upper; p++ )
            {
                // Make reference to particle
                const Particle& oldParticle = prevParticles[p];

                // tooFar might not be necessary if I cull beforehand
                bool tooFar = ( sliceIdx - oldParticle.ixLastSlice != 1 ); // Contiguity
                bool tooMany = ( oldParticle.nContributingSlices > maxStackCount );    // Count
                bool alreadyDone = ( oldParticle.pState == Particle::State::SEVER );    // Severed
                if ( tooFar || tooMany || alreadyDone )
                    continue;

                // See if the particle is within our range
                float dX = oldParticle.x - newParticle.x;
                float dY = oldParticle.y - newParticle.y;
                float distSq = powf( dX, 2 ) + powf( dY, 2 );
                if ( distSq < neighborRadius * neighborRadius )
                {
                    // If there already was a match, see if this one is better
                    if ( pBestMatch )
                    {
                        // Find the old distance
                        dX = pBestMatch->x - newParticle.x;
                        dY = pBestMatch->y - newParticle.y;

                        // If this one is closer, assign it as the match
                        if ( powf( dX, 2 ) + powf( dY, 2 ) > distSq )
                            pBestMatch = &prevParticles[p];
                    }
                    // No existing match, take this one
                    else
                        pBestMatch = &prevParticles[p];
                }
            }
        }

        // Could check sever state here
        return (Particle *) pBestMatch;
    }
};

// This gets called on matched particles and handles intensity state logic
// You should ensure this is thread safe beforehand, somehow (remove duplicates? not really sure)
struct UpdateMatchedParticle
{
    // Store the slice index - this should increase every time this
    // function is called to ensure we get the full particle picture
    int sliceIdx;
    UpdateMatchedParticle( int s ) : sliceIdx( s ) {}

    // This kind of thing could be parallelized in a smarter way, probably
    template <typename tuple_t>
    __host__ __device__
        int operator()( const tuple_t T )
    {
        // Function is called on a matched pair - new has been
        // been matched with old. See what that means for old. 
        Particle newParticle = thrust::get<0>( T );
        Particle * oldParticle = thrust::get<1>( T );
        switch ( oldParticle->pState )
        {
            // Shouldn't ever get no match, but assign the state and fall through
            case Particle::State::NO_MATCH:
                oldParticle->pState = Particle::State::INCREASING;
            // If we're increasing, see if the new guy prompts a decrease,
            // implying that we've passed through the center of the particle (in Z)
            // Should we check to see if more than one particle has contributed?
            case Particle::State::INCREASING:
                if ( oldParticle->i > newParticle.i )
                {
                    oldParticle->pState = Particle::State::DECREASING;
                    oldParticle->ixCenterSlice = sliceIdx;
                }
                // Otherwise see if we should update the peak intensity and z position
                else if ( newParticle.i > oldParticle->peakIntensity )
                {
                    oldParticle->peakIntensity = newParticle.i;
                    oldParticle->z = (float) sliceIdx;
                }
                break;
            // In this case, if it's still decreasing then fall through
            case Particle::State::DECREASING:
                if ( oldParticle->i > newParticle.i )
                    break;
                // We were decreasing, and now we're increasing again - cut it off
                oldParticle->pState = Particle::State::SEVER;
            //  Particle is severed - null it out for the code below
            case Particle::State::SEVER:
                oldParticle = nullptr;
        }

        // If we didn't sever and null out above
        // could do this in yet another call, if you were so inclined
        if ( oldParticle != nullptr )
        {
            // We're still getting information about this particle
            oldParticle->nContributingSlices++;
            oldParticle->ixLastSlice = sliceIdx;

            // I don't know about the averaged position thing
            oldParticle->x = 0.5f * ( oldParticle->x + newParticle.x );
            oldParticle->y = 0.5f * ( oldParticle->y + newParticle.y );
        }

        // This gets passed to a discard iterator - doesn't matter what we return
        return 0;
    }
};

// Used to sort particles by their grid index
// This could be phased out if I could get sort to work with a transform iterator
struct ParticleOrderingComp
{
    int N, M;
    ParticleOrderingComp( int n, int m ) : N( n ), M( m ) {}

    __host__ __device__
        bool operator()( const Particle a, const Particle b )
    {
        return pixelToGridIdx( a, N, M ) < pixelToGridIdx( b, N, M );
    }
};

// We consider a particle "Found" if it's in the Severed intensity state and
// has at least 3 particles contributing (should the 3 be hardcoded?)
struct IsFoundParticle
{
    __host__ __device__
        bool operator()( const Particle p )
    {
        return p.pState == Particle::State::SEVER && p.nContributingSlices > 2;
    }
};

struct IsFoundParticleInSlice
{
    int ixSlice;
    IsFoundParticleInSlice(int _ixSlice) :
        ixSlice(_ixSlice)
    {}
    __host__ __device__
        bool operator()(const Particle p)
    {
        return p.ixCenterSlice == ixSlice && p.pState == Particle::State::SEVER && p.nContributingSlices > 2;
    }
};

// Constructor and destructor must be implemented here
ParticleFinder::Solver::Solver():
    m_upSolverImpl( new impl() )
{}

ParticleFinder::Solver::~Solver()
{}

int ParticleFinder::Solver::FindParticlesInImage( int nSliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, bool bLinkParticles /*= true*/, std::vector<FoundParticle> * pParticlesInImg /*= nullptr*/ )
{
    return m_upSolverImpl->FindParticlesInImage( nSliceIdx, d_Input, d_FilteredImage, d_ThreshImg, d_ParticleImg, bLinkParticles, pParticlesInImg );
}

ParticleFinder::Solver::impl::impl() :
    m_uMaskRadius( 3 ),
    m_uFeatureRadius( 6 ),
    m_uMinSliceCount( 3 ),
    m_uMaxSliceCount( 5 ),
    m_uNeighborRadius( 5 ),
    m_uMaxLevel( 3 ),
    m_uCurPrevParticleCount( 0 )
{}

void ParticleFinder::Solver::impl::Init()
{
    // Reset all internal members
    // *this = impl();

    // Neighbor region diameter
    int diameter = 2 * m_uMaskRadius + 1;
    cv::Size maskSize( diameter, diameter );

    // Make host mats
    cv::Mat h_Circ( maskSize, CV_32F, 0.f );
    cv::Mat h_RX( maskSize, CV_32F, 0.f );
    cv::Mat h_RY( maskSize, CV_32F, 0.f );
    cv::Mat h_R2( maskSize, CV_32F, 0.f );

    // set up circle mask
    cv::circle( h_Circ, cv::Point( m_uMaskRadius, m_uMaskRadius ), m_uMaskRadius, 1.f, -1 );

    // set up Rx and part of r2
    for ( int y = 0; y < diameter; y++ )
    {
        for ( int x = 0; x < diameter; x++ )
        {
            cv::Point p( x, y );
            h_RX.at<float>( p ) = x + 1;
            h_RY.at<float>( p ) = y + 1;
            h_R2.at<float>( p ) = pow( -(float) m_uMaskRadius + x, 2 ) + pow( -(float) m_uMaskRadius + y, 2 );
        }
    }

    // I forget what these do...
    cv::threshold( h_R2, h_R2, pow( (double) m_uMaskRadius, 2 ), 1, cv::THRESH_TOZERO_INV );
    cv::multiply( h_RX, h_Circ, h_RX );
    cv::multiply( h_RY, h_Circ, h_RY );

    // For host debugging
#if SOLVER_DEVICE
    // Upload to continuous gpu mats
    m_dCircleMask = GetContinuousGpuMat( h_Circ );
    m_dRadXKernel = GetContinuousGpuMat( h_RX );
    m_dRadYKernel = GetContinuousGpuMat( h_RY );
    m_dRadSqKernel = GetContinuousGpuMat( h_R2 );
#else
    h_Circ.copyTo( m_dCircleMask );
    h_RX.copyTo( m_dRadXKernel );
    h_RY.copyTo( m_dRadYKernel );
    h_R2.copyTo( m_dRadSqKernel );
#endif
}

// This function removes particles from the vector of previously found particles if they
// pass the predicate MaybeRemoveParticle
size_t ParticleFinder::Solver::impl::cullExistingParticles( int curSliceIdx )
{
    size_t u_preremovePrevParticleCount = m_uCurPrevParticleCount;
    auto itLastPrevParticleEnd = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;
    auto itCurPrevParticleEnd = thrust::remove_if( m_dPrevParticleVec.begin(), itLastPrevParticleEnd, MaybeRemoveParticle( curSliceIdx, m_uMinSliceCount ) );
    m_uCurPrevParticleCount = itCurPrevParticleEnd - m_dPrevParticleVec.begin();
    size_t nRemovedParticles = u_preremovePrevParticleCount - m_uCurPrevParticleCount;

    return nRemovedParticles;
}

// Given the processed particle image, this function finds the particle locations and returns a vector of Particle objects
ParticleFinder::Solver::impl::ParticleVec ParticleFinder::Solver::impl::findNewParticles( UcharVec& d_ParticleImgVec, Floatptr pThreshImg, int N, int sliceIdx )
{
    // Create pointers to our kernels
    Floatptr d_pCirleKernel( (float *) m_dCircleMask.data );
    Floatptr d_pRxKernel( (float *) m_dRadXKernel.data );
    Floatptr d_pRyKernel( (float *) m_dRadYKernel.data );
    Floatptr d_pR2Kernel( (float *) m_dRadSqKernel.data );

    // For each pixel in the particle image, we care if it's nonzero and if it's far enough from the edges
    // So we need its index (transformable into twoD pos) and its value, which we zip
    auto itDetectParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.begin(), thrust::counting_iterator<int>( 0 ) ) );
    auto itDetectParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_ParticleImgVec.end(), thrust::counting_iterator<int>( N*N ) ) );

    // Then, if the particle fits our criteria, we copy its index (from the counting iterator) into this vector, and discard the uchar
    IntVec d_NewParticleIndicesVec( N*N );
    auto itFirstNewParticle = thrust::make_zip_iterator( thrust::make_tuple( thrust::discard_iterator<>(), d_NewParticleIndicesVec.begin() ) );
    auto itLastNewParticle = thrust::copy_if( itDetectParticleBegin, itDetectParticleEnd, itFirstNewParticle, IsParticleAtIdx( N, m_uFeatureRadius ) );
    size_t newParticleCount = itLastNewParticle - itFirstNewParticle;

    // Now transform each index into a particle by looking at values inside the lmimg and using the kernels
    ParticleVec d_NewParticleVec( newParticleCount );
    thrust::transform( d_NewParticleIndicesVec.begin(), d_NewParticleIndicesVec.begin() + newParticleCount, d_NewParticleVec.begin(),
#if SOLVER_DEVICE
                       MakeParticleFromIdx( sliceIdx, N, m_uMaskRadius, pThreshImg.get(), d_pCirleKernel.get(), d_pRxKernel.get(), d_pRyKernel.get(), d_pR2Kernel.get() ) );
#else
                       MakeParticleFromIdx(sliceIdx, N, m_uMaskRadius, pThreshImg, d_pCirleKernel, d_pRxKernel, d_pRyKernel, d_pR2Kernel) );
#endif

    return d_NewParticleVec;
}

// This function recreates the grid cell ranges given the current container of previous particles
void ParticleFinder::Solver::impl::createGridCells( int N )
{
    // We don't bother if there are no previous particles
    if ( m_dPrevParticleVec.empty() )
        return;

    // If our grid cell vectors are empty, create them now
    if ( m_dGridCellLowerBoundVec.empty() || m_dGridCellUpperBoundVec.empty() )
    {
        const int cellSize = N >> m_uMaxLevel; // 1D size of cell in pixels
        const int cellCount = N / cellSize;    // 1D count of cells

        // Create 2D cell sub-image
        const int nTotalCells = cellCount * cellCount;
        m_dGridCellLowerBoundVec.resize( nTotalCells );
        m_dGridCellUpperBoundVec.resize( nTotalCells );
    }

    // Some typedefs, we use a transform iterator to convert particles into indices
    using particleIter = ParticleVec::iterator;
    using pixelToGridIdxIter = thrust::transform_iterator < PixelToGridIdx, particleIter >;

    // Create an iterator to the end of our current previous particle container (might not be m_dPrevParticleVec.end())
    auto itCurPrevParticleEnd = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;

    // Create the transform iterator that iterates over our previous particles and returns their grid indices
    pixelToGridIdxIter itPrevParticleBegin = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( m_dPrevParticleVec.begin(), PixelToGridIdx( N, m_uMaxLevel ) );
    pixelToGridIdxIter itPrevParticleEnd = thrust::make_transform_iterator<PixelToGridIdx, particleIter>( itCurPrevParticleEnd, PixelToGridIdx( N, m_uMaxLevel ) );

    // Find the ranges of previous particless
    const size_t nTotalCells = m_dGridCellLowerBoundVec.size();
    thrust::lower_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), m_dGridCellLowerBoundVec.begin() );
    thrust::upper_bound( itPrevParticleBegin, itPrevParticleEnd, thrust::counting_iterator<int>( 0 ), thrust::counting_iterator<int>( nTotalCells ), m_dGridCellUpperBoundVec.begin() );
}

// For each new particle, given the range of previous particles to search through, find the best match and return a pointer to its address
// If the pointer is null, then no match was found
ParticleFinder::Solver::impl::ParticlePtrVec ParticleFinder::Solver::impl::findParticleMatches( ParticleVec& d_NewParticleVec, int N, int sliceIdx )
{
    ParticlePtrVec d_ParticleMatchVec( d_NewParticleVec.size(), (Particle *)nullptr );

    // Only go through this is there are cells we could match with
    if ( m_dPrevParticleVec.empty() == false )
        thrust::transform( d_NewParticleVec.begin(), d_NewParticleVec.end(), d_ParticleMatchVec.begin(),
#if SOLVER_DEVICE
        ParticleMatcher( N, m_uMaxLevel, sliceIdx, m_uMaxSliceCount, m_dGridCellLowerBoundVec.size(), m_uNeighborRadius, m_dGridCellLowerBoundVec.data().get(), m_dGridCellUpperBoundVec.data().get(), m_dPrevParticleVec.data().get() ) );
#else
        ParticleMatcher(N, m_uMaxLevel, sliceIdx, m_uMaxSliceCount, m_dGridCellLowerBoundVec.size(), m_uNeighborRadius, m_dGridCellLowerBoundVec.data(), m_dGridCellUpperBoundVec.data(), m_dPrevParticleVec.data()) );
#endif

    return d_ParticleMatchVec;
}

// For every matched particle, update its intensity state / position
void ParticleFinder::Solver::impl::updateMatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec, int sliceIdx )
{
    // Zip the pointer vec and newparticle vec
    auto itNewParticleToMatchedParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.begin(), d_ParticleMatchVec.begin() ) );
    auto itNewParticleToMatchedParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.end(), d_ParticleMatchVec.end() ) );

    // If there was a match, update the intensity state. I don't know how to do a for_each_if other than a transform_if that discards the output
    thrust::transform_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, thrust::discard_iterator<>(), UpdateMatchedParticle( sliceIdx ), CheckIfMatchIsNotNull() );

#if _DEBUG
    // Useful for me to know how these start to spread out on debug
    auto itCurPrevParticleEnd = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;
    size_t numInNoMatch = thrust::count_if(m_dPrevParticleVec.begin(), itCurPrevParticleEnd, IsParticleInState(Particle::State::NO_MATCH));
    size_t numInIncreasing = thrust::count_if( m_dPrevParticleVec.begin(), itCurPrevParticleEnd,IsParticleInState(Particle::State::INCREASING));
    size_t numInDecreasing = thrust::count_if( m_dPrevParticleVec.begin(), itCurPrevParticleEnd,IsParticleInState(Particle::State::DECREASING));
    size_t numInSever = thrust::count_if( m_dPrevParticleVec.begin(), itCurPrevParticleEnd, IsParticleInState(Particle::State::SEVER));
#endif
}

// For the particles that weren't matched, stream compact them into a vector and return it
ParticleFinder::Solver::impl::ParticleVec ParticleFinder::Solver::impl::consolidateUnmatchedParticles( ParticleVec& d_NewParticleVec, ParticlePtrVec& d_ParticleMatchVec )
{
    // Zip the pointer vec and newparticle vec
    auto itNewParticleToMatchedParticleBegin = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.begin(), d_ParticleMatchVec.begin() ) );
    auto itNewParticleToMatchedParticleEnd = thrust::make_zip_iterator( thrust::make_tuple( d_NewParticleVec.end(), d_ParticleMatchVec.end() ) );

    // Copy all unmatched particles into a new vector; we copy a tuple of new particles and pointers to matches, discarding the pointers
    ParticleVec d_UnmatchedParticleVec( d_NewParticleVec.size() );
    auto itNewParticleAndPrevParticleMatchBegin = thrust::make_zip_iterator( thrust::make_tuple( d_UnmatchedParticleVec.begin(), thrust::discard_iterator<>() ) );

    // Copy new particles if their corresponding match is null
    auto itNewParticleAndPrevParticleMatchEnd = thrust::copy_if( itNewParticleToMatchedParticleBegin, itNewParticleToMatchedParticleEnd, itNewParticleAndPrevParticleMatchBegin, CheckIfMatchIsNull() );
    size_t numUnmatchedParticles = itNewParticleAndPrevParticleMatchEnd - itNewParticleAndPrevParticleMatchBegin;

    // Size down and return
    d_UnmatchedParticleVec.resize( numUnmatchedParticles );
    return d_UnmatchedParticleVec;
}

// Given our previous particles and the newly found unmatched particles, merge them into a sorted container
void ParticleFinder::Solver::impl::mergeUnmatchedParticles( ParticleVec& d_UnmatchedParticleVec, int N )
{
    // We have two options here; the easy option is to just tack these new particles onto the previous particle vector and sort the whole thing
    // alternatively you could set a flag in previous particles if the matching process caused them to move to a new grid cell and then treat those particles as unmatched
    // you could then sort the unmatched particles (relatively few compared to the count of previous particles) and then merge them into the prev particle vec, which is still sorted
    // Below is the first option, whihc was easier.

    // first make room for the new particles, if we need it
    size_t newPrevParticleCount = d_UnmatchedParticleVec.size() + m_uCurPrevParticleCount;
    if ( newPrevParticleCount > m_dPrevParticleVec.size() )
        m_dPrevParticleVec.resize( newPrevParticleCount );

    // copy unmatched particles onto the original end of the previous particle vec
    auto itNewParticleDest = m_dPrevParticleVec.begin() + m_uCurPrevParticleCount;
    auto itEndOfPrevParticles = thrust::copy( d_UnmatchedParticleVec.begin(), d_UnmatchedParticleVec.end(), itNewParticleDest );

    // Sort the whole thing
    thrust::sort( m_dPrevParticleVec.begin(), itEndOfPrevParticles, ParticleOrderingComp( N, m_uMaxLevel ) );
    m_uCurPrevParticleCount = newPrevParticleCount;
}

int ParticleFinder::Solver::impl::FindParticlesInImage( int nSliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, bool bLinkParticles, std::vector<FoundParticle> * pParticlesInImg )
{
    // Make sure we've initialized something
    if (m_dRadSqKernel.empty())
        return 0;

    // We assume the row and column dimensions are equal
    const int N = d_Input.rows;

    // Make a device vector out of the particle buffer pointer (it's contiguous)
#if SOLVER_DEVICE
    UcharPtr d_pParticleImgBufStart((unsigned char *)d_ParticleImg.datastart);
    UcharPtr d_pParticleImgBufEnd((unsigned char *)d_ParticleImg.dataend);
    UcharVec d_ParticleImgVec(d_pParticleImgBufStart, d_pParticleImgBufEnd);
    Floatptr d_pThreshImgBuf((float *)d_ThreshImg.data);
#else
    // For host debugging
    cv::Mat h_ThreshImg;
    d_ThreshImg.download(h_ThreshImg);
    Floatptr d_pThreshImgBuf(h_ThreshImg.ptr<float>());

    Floatptr d_pLocalMaxImgBuf(h_ThreshImg.ptr<float>());
    thrust::device_vector<unsigned char> d_vData((unsigned char *)d_ParticleImg.datastart, (unsigned char *)d_ParticleImg.dataend);
    UcharVec d_ParticleImgVec = d_vData;
#endif

    // Cull the herd
    size_t numParticlesRemoved = cullExistingParticles( nSliceIdx );

    // Find new particles
    ParticleVec d_NewParticleVec = findNewParticles( d_ParticleImgVec, d_pThreshImgBuf, N, nSliceIdx );
    size_t numParticlesFound = d_NewParticleVec.size();

    // Store new particles if requested
    if ( pParticlesInImg )
    {
        // Convert all found particles to the foundparticle type
        // Reserve enough space for all previously found, though the actual may be less
        FoundParticleVec d_vRet( numParticlesFound );
        pParticlesInImg->resize( numParticlesFound ); // (needed?)

        // We consider a particle "found" if it's intensity state is severed
        // Transform all found particles that meet the IsFoundParticle criterion into FoundParticles
        auto itFoundParticleEnd = thrust::transform( d_NewParticleVec.begin(), d_NewParticleVec.end(), d_vRet.begin(), Particle2FoundParticle() );
        auto nParticles = itFoundParticleEnd - d_vRet.begin();

        // Download to host
        thrust::copy( d_vRet.begin(), d_vRet.end(), pParticlesInImg->begin() );
    }

    // Get out now if they don't want us to link particles
    // Return the number of particles found in this image
    if (bLinkParticles == false)
        return (int)numParticlesFound;

    // Initialize grid cells given current container of previous particles
    createGridCells( N );

    // Tranform new particles into a vector of particle pointers; if they are null then no match was found (?)
    ParticlePtrVec d_ParticleMatchVec = findParticleMatches( d_NewParticleVec, N, nSliceIdx );

    // For particles we were able to match, update their intensity states
    updateMatchedParticles( d_NewParticleVec, d_ParticleMatchVec, nSliceIdx );

    // Copy all unmatched particles into a new vector; we copy a tuple of new particles and pointers to matches, discarding the pointers
    ParticleVec d_UnmatchedParticleVec = consolidateUnmatchedParticles( d_NewParticleVec, d_ParticleMatchVec );

    // Merge unmatched particles into our container, preserving grid index order
    mergeUnmatchedParticles( d_UnmatchedParticleVec, N );

#if _DEBUG
    std::cout << "Slice Idx:\t" << nSliceIdx << "\tNew Particles:\t" << numParticlesFound << "\tUnmatched Particles:\t" << d_UnmatchedParticleVec.size() << "\tFound Particles:\t" << m_uCurPrevParticleCount << "\tCulled Particles:\t" << numParticlesRemoved << std::endl;
#endif

    return m_uCurPrevParticleCount;
}

std::vector<ParticleFinder::FoundParticle> ParticleFinder::Solver::impl::GetFoundParticles() const
{
    // Convert all found particles to the foundparticle type
    // Reserve enough space for all previously found, though the actual may be less
    ParticleVec d_vFoundParticles( m_uCurPrevParticleCount );

    // We consider a particle "found" if it's intensity state is severed
    auto itParticleEnd = thrust::copy_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), d_vFoundParticles.begin(), IsFoundParticle() );
    if ( itParticleEnd == d_vFoundParticles.begin() )
        return{};

    int nParticles = itParticleEnd - d_vFoundParticles.begin();

    //

    // Transform all found particles that meet the IsFoundParticle criterion into FoundParticles
    FoundParticleVec d_vTransformedFoundParticles( nParticles );
    thrust::transform( d_vFoundParticles.begin(), itParticleEnd, d_vTransformedFoundParticles.begin(), Particle2FoundParticle());

    // Download to host, return
    std::vector<ParticleFinder::FoundParticle> vRet( nParticles );
    thrust::copy( d_vTransformedFoundParticles.begin(), d_vTransformedFoundParticles.end(), vRet.begin());

    return vRet;
}

std::vector<ParticleFinder::FoundParticle> ParticleFinder::Solver::impl::GetFoundParticlesInSlice(int ixSlice) const
{
    // Convert all found particles to the foundparticle type
    // Reserve enough space for all previously found, though the actual may be less
    ParticleVec d_vFoundParticles( m_uCurPrevParticleCount );

    // We consider a particle "found" if it's intensity state is severed
    auto itParticleEnd = thrust::copy_if( m_dPrevParticleVec.begin(), m_dPrevParticleVec.end(), d_vFoundParticles.begin(), IsFoundParticleInSlice(ixSlice) );
    if ( itParticleEnd == d_vFoundParticles.begin() )
        return{};

    int nParticles = itParticleEnd - d_vFoundParticles.begin();

    //

    // Transform all found particles that meet the IsFoundParticle criterion into FoundParticles
    FoundParticleVec d_vTransformedFoundParticles( nParticles );
    thrust::transform( d_vFoundParticles.begin(), itParticleEnd, d_vTransformedFoundParticles.begin(), Particle2FoundParticle() );

    // Download to host, return
    std::vector<ParticleFinder::FoundParticle> vRet( nParticles );
    thrust::copy( d_vTransformedFoundParticles.begin(), d_vTransformedFoundParticles.end(), vRet.begin() );

    return vRet;
}

std::vector<ParticleFinder::FoundParticle> ParticleFinder::Solver::GetFoundParticles() const
{
    return m_upSolverImpl->GetFoundParticles();
}

std::vector<ParticleFinder::FoundParticle> ParticleFinder::Solver::GetFoundParticlesInSlice(int ixSlice) const
{
    return m_upSolverImpl->GetFoundParticlesInSlice(ixSlice);
}

void ParticleFinder::Solver::Init()
{
    return m_upSolverImpl->Init();
}

void ParticleFinder::Solver::Reset()
{
    m_upSolverImpl.reset( new impl() );
}

void ParticleFinder::Solver::SetMaskRadius( int mR )
{
    m_upSolverImpl->m_uMaskRadius = mR;
}

void ParticleFinder::Solver::SetFeatureRadius( int fR )
{
    m_upSolverImpl->m_uFeatureRadius = fR;
}

void ParticleFinder::Solver::SetMinSliceCount( int nMinSC )
{
    m_upSolverImpl->m_uMinSliceCount = nMinSC;
}

void ParticleFinder::Solver::SetMaxSliceCount( int nMaxSC )
{
    m_upSolverImpl->m_uMaxSliceCount = nMaxSC;
}

void ParticleFinder::Solver::SetNeighborRadius( int nR )
{
    m_upSolverImpl->m_uNeighborRadius = nR;
}

void ParticleFinder::Solver::SetMaxLevel( int mL )
{
    m_upSolverImpl->m_uMaxLevel = mL;
}

int ParticleFinder::Solver::GetMaskRadius() const
{
    return m_upSolverImpl->m_uMaskRadius;
}

int ParticleFinder::Solver::GetFeatureRadius() const
{
    return m_upSolverImpl->m_uFeatureRadius;
}

int ParticleFinder::Solver::GetMinSliceCount() const
{
    return m_upSolverImpl->m_uMinSliceCount;
}

int ParticleFinder::Solver::GetMaxSliceCount() const
{
    return m_upSolverImpl->m_uMaxSliceCount;
}

int ParticleFinder::Solver::GetNeighborRadius() const
{
    return m_upSolverImpl->m_uNeighborRadius;
}

int ParticleFinder::Solver::GetMaxLevel() const
{
    return m_upSolverImpl->m_uMaxLevel;
}