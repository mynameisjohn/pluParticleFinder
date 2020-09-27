#pragma once

#include "ParticleFinder.h"

#include <thrust/device_vector.h>

#ifndef SOLVER_DEVICE 
    #define SOLVER_DEVICE 0
#endif
#if SOLVER_DEVICE
    #define thrust_operator __device__
    #define thrust_exec thrust::device
#else
    #define thrust_operator __host__
    #define thrust_exec thrust::host
#endif

// Our device particle struct
struct Particle
{
    float x{ 0 };                        // X pos
    float y{ 0 };                        // Y pos
    float z{ 0 };                        // Z pos
    float i{ 0 };                        // Intensity
    float r2{ 0 };                       // Radius
    Particle* parent{ nullptr };
    Particle* match{ nullptr };
    int nContributingSlices{ 0 };        // # of slices contributing
};

    // Useful typedefs of mine
#if SOLVER_DEVICE
    using UcharVec = thrust::device_vector < unsigned char >;
    using UcharPtr = thrust::device_ptr < unsigned char >;
    using IntVec = thrust::device_vector < int >;
    using IntPtr = thrust::device_ptr < int >;
    using FloatVec = thrust::device_vector < float >;
    using Floatptr = thrust::device_ptr < float >;
    using ParticleVec = thrust::device_vector < Particle >;
    using ParticlePtrVec = thrust::device_vector < Particle* >;
    using FoundParticleVec = thrust::device_vector <ParticleFinder::FoundParticle>;
    using Img = cv::cuda::GpuMat;
#else
    // I use these for host debugging
    using UcharVec = thrust::host_vector < unsigned char >;
    using UcharPtr = unsigned char*;
    using IntVec = thrust::host_vector < int >;
    using IntPtr = int*;
    using FloatVec = thrust::host_vector < float >;
    using Floatptr = float*;
    using ParticleVec = thrust::host_vector < Particle >;
    using ParticlePtrVec = thrust::host_vector < Particle* >;
    using FoundParticleVec = thrust::host_vector <ParticleFinder::FoundParticle>;
    using Img = cv::Mat;
#endif

// Solver implementation
struct ParticleFinder::Solver::impl
{
    // sets default parameters
    impl ();

    // implementations of Solver functions
    void Init (int N, int threadCount);
    int FindParticlesInImage (int thraedNum, int stackNum, int sliceIdx, GpuMat d_Input, GpuMat d_FilteredImage, GpuMat d_ThreshImg, GpuMat d_ParticleImg, std::vector<FoundParticle>* pParticlesInImg);
    std::map<int, std::vector<FoundParticle>> LinkFoundParticles ();

    // Particle Finding params
    int _maskRadius;                   // The radius of our particle mask kernels
    int _featureRadius;                // The radius within the image we'd like to consier
    
    // Particle Linking params
    int _minSliceCount;                // The minimum # of slices we require to contribute to a particle
    int _maxSliceCount;                // The maximum # of slices we allow to contribute to a particle
    float _neighborRadius;             // The radius in which we search for new particles
    float _xyFactor;
    float _zFactor;

    Img _circleMask;                   // The circle mask, just a circle of radius m_uMaskRadius, each value is 1
    Img _radXKernel;                   // The x circle mask, used to calculate an offset to the x coordinate
    Img _radYKernel;                   // The y circle mask, used to calculate an offset to the y coordinate
    Img _radSqKernel;                  // The r2 circle mask, used to calculate particle radius

    Floatptr _circleKernelPtr;
    Floatptr _radXKernelPtr;
    Floatptr _radYKernelPtr;
    Floatptr _radSqKernelPtr;

    std::vector<IntVec> _newIndicesVec;

    // used to map slices to found particles
    std::vector<std::map<int, std::map<int, ParticleVec>>> _stackToParticles;

    // used for 
    const static int MASK_RAD_MAX = 7;
};