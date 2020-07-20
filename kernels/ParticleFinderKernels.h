#pragma once

#include "ParticleSolver.h"

struct IsFoundParticle
{
    thrust_operator bool operator()(const Particle p)
    {
        return p.parent == nullptr;
    }
};
struct IsNotFoundParticle
{
    thrust_operator bool operator()(const Particle p)
    {
        return p.parent != nullptr;
    }
};
// apply to parents to get proper initial values
struct initFoundParticles
{
    float _xyFactor;
    initFoundParticles (float xyFactor) : _xyFactor (xyFactor) {}

    thrust_operator int operator()(Particle& p)
    {
        p.x *= p.i;
        p.y *= p.i;
        p.z *= p.i;

        p.r2 *= _xyFactor;

        p.nContributingSlices = 1;

        return 0;
    }
};
// apply to children to reduce into parent
struct computeAverageSum
{
    float _xyFactor;
    computeAverageSum (float xyFactor) : _xyFactor (xyFactor) {}

    thrust_operator int operator()(Particle p)
    {
#if SOLVER_DEVICE
        atomicAdd (&p.parent->x, p.i * p.x);
        atomicAdd (&p.parent->y, p.i * p.y);
        atomicAdd (&p.parent->z, p.i * p.z);
        atomicAdd (&p.parent->i, p.i);

        // https://stackoverflow.com/a/51549250/1973454
        // (I took the positive version of this because the radius is always positive
        atomicMax ((int*)&p.parent->r2, __float_as_int (_xyFactor * p.r2));

        atomicAdd (&p.parent->nContributingSlices, 1);
#else
        p.parent->x += p.i * p.x;
        p.parent->y += p.i * p.y;
        p.parent->z += p.i * p.z;

        p.parent->r2 = fmaxf (_xyFactor * p.r2, p.parent->r2);

        p.parent->i += p.i;
        p.parent->nContributingSlices++;
#endif

        return 0;
    }
};
// apply to parent to get averaged positions
struct averageParticlePositions
{
    float _xyFactor;
    float _zFactor;
    averageParticlePositions (float xyFactor, float zFactor)
    : _xyFactor (xyFactor),
      _zFactor (zFactor)
    {}


    thrust_operator int operator()(Particle& p)
    {
        p.x *= _xyFactor / p.i;
        p.y *= _xyFactor / p.i;
        p.z *= _zFactor / p.i;
        // p.i /= float (p.nContributingSlices); // does this need to be averaged?

        return 0;
    }
};
// apply to parent to get averaged positions
struct CheckParticleBoundaries
{
    float _minX;
    float _minY;
    float _minZ;
    float _maxX;
    float _maxY;
    float _maxZ;

    thrust_operator bool operator()(const Particle p)
    {
        return
            (p.x < _minX) || (p.x > _maxX) ||
            (p.y < _minY) || (p.y > _maxY) ||
            (p.z < _minZ) || (p.z > _maxZ);
    }
};
struct FilterParticlesBySliceCount
{
    int _minSlices;
    thrust_operator bool operator()(const Particle P)
    {
        return P.nContributingSlices < _minSlices;
    }
};
template<const int N>
struct MinVectorElement
{
    thrust_operator bool operator()(const Particle A, const Particle B) const
    {
        const float* aPos = &A.x;
        const float* bPos = &B.x;
        return aPos[N] < bPos[N];
    }
};
struct Particle2FoundParticle
{
    thrust_operator ParticleFinder::FoundParticle operator()(const Particle& p)
    {
        ParticleFinder::FoundParticle ret;
        ret.fPosX = p.x;
        ret.fPosY = p.y;
        ret.fPosZ = p.z;
        ret.fIntensity = p.i;
        ret.fR2 = p.r2;
        return ret;
    }
};

// Particle detection predicate
struct IsParticleAtIdx
{
    int N;            // Image dimension
    int featureRad;    // Image feature radius

    IsParticleAtIdx (int n, int k) : N (n), featureRad (k) {}

    template <typename tuple_t>
    thrust_operator bool operator()(tuple_t T)
    {
        // Unpack tuple
        unsigned char val = thrust::get<0> (T);
        int idx = thrust::get<1> (T);

        // get xy coords from pixel index
        int x = idx % N;
        int y = idx / N;

        // We care if the pixel is nonzero and its within the kernel radius
        return (val != 0) && (x > featureRad) && (y > featureRad) && (x + featureRad < N) && (y + featureRad < N);
    }
};// Turn an index at which we've detected a particle (above) into a real particle
struct MakeParticleFromIdx
{
    int stackNum;      // Current stack number
    int sliceIdx;      // Current slice index
    int kernelRad;     // Kernel (mask) radius
    int N;             // Image dimension
    float* lmImg;      // pointer to image in which particle was detected
    float* circMask;   // pointer to circle mask
    float* rxMask;     // pointer to x offset mask
    float* ryMask;     // pointer to y offset mask
    float* rSqMask;    // pointer to r2 mask 
    MakeParticleFromIdx (int nStack, int sIdx, int n, int kRad, float* lm, float* cK, float* xK, float* yK, float* sqK) :
        stackNum (nStack),
        sliceIdx (sIdx),
        N (n),
        kernelRad (kRad),
        lmImg (lm),
        circMask (cK),
        rxMask (xK),
        ryMask (yK),
        rSqMask (sqK)
    {}

    thrust_operator Particle operator()(int idx)
    {
        // Grab x, y values
        int x = idx % N;
        int y = idx / N;

        // Make tmp pointers to our masks and advance them
        // as we iterate to perform the multiplication
        float* tmpCircPtr = circMask;
        float* tmpXPtr = rxMask;
        float* tmpYPtr = ryMask;
        float* tmpR2Ptr = rSqMask;

        // To be calculated
        float total_mass (0);
        float x_offset (0), y_offset (0);
        float r2_sum (0);

        // Apply the mask as a multiplcation
        for (int iY = -kernelRad; iY <= kernelRad; iY++)
        {
            // For y, go down then up
            float* ptrY = &lmImg[idx + (N * iY)];
            for (int iX = -kernelRad; iX <= kernelRad; iX++)
            {
                // Get the local max img value
                float lmImgVal = ptrY[iX];

                // Multiply by mask value, sum, advance mask pointer
                total_mass += lmImgVal * (*tmpCircPtr++);
                x_offset += lmImgVal * (*tmpXPtr++);
                y_offset += lmImgVal * (*tmpYPtr++);
                r2_sum += lmImgVal * (*tmpR2Ptr++);
            }
        }

        // Calculate x val, y val
        // (in the original code the calculation is
        // x_val = x + x_offset/total_maxx - kernelRad - 1... not sure if I still need that)
        float total_mass_inv = 1.f / total_mass;
        float x_val = float (x) + x_offset * total_mass_inv - kernelRad - 1;
        float y_val = float (y) + y_offset * total_mass_inv - kernelRad - 1;
        float z_val = float (sliceIdx + 1);
        float r2_val = r2_sum * total_mass_inv;

        // Construct particle and return
        Particle p{ x_val, y_val, z_val, total_mass, r2_val };
        return p;
    }
};
struct SeverParticle
{
    thrust_operator int operator()(Particle& p)
    {
        p.parent = nullptr;
        p.nContributingSlices = 0;

        return 0;
    }
};
struct CheckParticleRadius
{
    Particle* prevParticles;
    Particle* curParticles;
    int prevParticleCount;
    float r2Max;

    thrust_operator bool operator()(int idx)
    {
        int ixPrev = idx % prevParticleCount;
        int ixCur = idx / prevParticleCount;
        Particle& prev = prevParticles[ixPrev];
        Particle& cur = curParticles[ixCur];

        float dX = prev.x - cur.x;
        float dY = prev.y - cur.y;
        float r2 = (dX * dX + dY * dY);
        return r2 < r2Max;
    }
};
struct AttachParticle
{
    Particle* prevParticles;
    Particle* curParticles;
    int prevParticleCount;

    thrust_operator int operator() (const int idx)
    {
        int ixPrev = idx % prevParticleCount;
        int ixCur = idx / prevParticleCount;
        Particle& prev = prevParticles[ixPrev];
        Particle& cur = curParticles[ixCur];

#if SOLVER_DEVICE
        unsigned long long int* parentAddr = (unsigned long long int*) & cur.parent;
        unsigned long long int newParent = (unsigned long long int)(prev.parent ? prev.parent : &prev);
        unsigned long long int* matchAddr = (unsigned long long int*) & cur.match;
        unsigned long long int newMatch = (unsigned long long int)(&prev);
        atomicExch (parentAddr, newParent);
        atomicExch (matchAddr, newMatch);
        atomicExch (&cur.nContributingSlices, prev.nContributingSlices + 1);
#else
        Particle* newParent = prev.parent ? prev.parent : &prev;
        cur.parent = prev.parent ? prev.parent : &prev;
        cur.nContributingSlices = prev.nContributingSlices + 1;
#endif

        return 0;
    }
};
struct ShouldSeverParticle
{
    int minSlices;
    int maxSlices;

    thrust_operator bool operator()(Particle p)
    {
        return ((p.nContributingSlices >= minSlices) && (p.i > p.match->i)) || (p.nContributingSlices >= maxSlices);
    }
};