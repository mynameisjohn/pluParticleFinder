//#include "ParticleFinderKernels.cuh"
//
//thrust_operator bool IsFoundParticle::operator()(const Particle p)
//{
//    return p.parent == nullptr;
//}
//thrust_operator bool IsNotFoundParticle::operator()(const Particle p)
//{
//    return p.parent != nullptr;
//}
//// apply to parents to get proper initial values
//thrust_operator int initFoundParticles::operator()(Particle& p)
//{
//    p.x *= p.i;
//    p.y *= p.i;
//    p.z *= p.i;
//
//    p.r2 *= _xyFactor;
//
//    p.nContributingSlices = 1;
//
//    return 0;
//}
//// apply to children to reduce into parent
//thrust_operator int computeAverageSum::operator()(Particle p)
//{
//#if SOLVER_DEVICE
//    atomicAdd (&p.parent->x, p.i * p.x);
//    atomicAdd (&p.parent->y, p.i * p.y);
//    atomicAdd (&p.parent->z, p.i * p.z);
//    atomicAdd (&p.parent->i, p.i);
//
//    // https://stackoverflow.com/a/51549250/1973454
//    // (I took the positive version of this because the radius is always positive
//    atomicMax ((int*)&p.parent->r2, __float_as_int (_xyFactor * p.r2));
//
//    atomicAdd (&p.parent->nContributingSlices, 1);
//#else
//    p.parent->x += p.i * p.x;
//    p.parent->y += p.i * p.y;
//    p.parent->z += p.i * p.z;
//
//    p.parent->r2 = fmaxf (_xyFactor * p.r2, p.parent->r2);
//
//    p.parent->i += p.i;
//    p.parent->nContributingSlices++;
//#endif
//
//    return 0;
//}
//// apply to parent to get averaged positions
//thrust_operator int averageParticlePositions::operator()(Particle& p)
//{
//    p.x *= _xyFactor / p.i;
//    p.y *= _xyFactor / p.i;
//    p.z *= _zFactor / p.i;
//    // p.i /= float (p.nContributingSlices); // does this need to be averaged?
//
//    return 0;
//}
//// apply to parent to get averaged positions
//thrust_operator bool CheckParticleBoundaries::operator()(const Particle p)
//{
//    return
//        (p.x < _minX) || (p.x > _maxX) ||
//        (p.y < _minY) || (p.y > _maxY) ||
//        (p.z < _minZ) || (p.z > _maxZ);
//}
//thrust_operator bool FilterParticlesBySliceCount::operator()(const Particle P)
//{
//    return P.nContributingSlices < _minSlices;
//}
//thrust_operator ParticleFinder::FoundParticle Particle2FoundParticle::operator()(const Particle& p)
//{
//    ParticleFinder::FoundParticle ret;
//    ret.fPosX = p.x;
//    ret.fPosY = p.y;
//    ret.fPosZ = p.z;
//    ret.fIntensity = p.i;
//    ret.fR2 = p.r2;
//    return ret;
//}
//
//MakeParticleFromIdx::MakeParticleFromIdx (int nStack, int sIdx, int n, int kRad, float* lm, float* cK, float* xK, float* yK, float* sqK) :
//    stackNum (nStack),
//    sliceIdx (sIdx),
//    N (n),
//    kernelRad (kRad),
//    lmImg (lm),
//    circMask (cK),
//    rxMask (xK),
//    ryMask (yK),
//    rSqMask (sqK)
//{}
//thrust_operator Particle MakeParticleFromIdx::operator()(int idx)
//{
//    // Grab x, y values
//    int x = idx % N;
//    int y = idx / N;
//
//    // Make tmp pointers to our masks and advance them
//    // as we iterate to perform the multiplication
//    float* tmpCircPtr = circMask;
//    float* tmpXPtr = rxMask;
//    float* tmpYPtr = ryMask;
//    float* tmpR2Ptr = rSqMask;
//
//    // To be calculated
//    float total_mass (0);
//    float x_offset (0), y_offset (0);
//    float r2_sum (0);
//
//    // Apply the mask as a multiplcation
//    for (int iY = -kernelRad; iY <= kernelRad; iY++)
//    {
//        // For y, go down then up
//        float* ptrY = &lmImg[idx + (N * iY)];
//        for (int iX = -kernelRad; iX <= kernelRad; iX++)
//        {
//            // Get the local max img value
//            float lmImgVal = ptrY[iX];
//
//            // Multiply by mask value, sum, advance mask pointer
//            total_mass += lmImgVal * (*tmpCircPtr++);
//            x_offset += lmImgVal * (*tmpXPtr++);
//            y_offset += lmImgVal * (*tmpYPtr++);
//            r2_sum += lmImgVal * (*tmpR2Ptr++);
//        }
//    }
//
//    // Calculate x val, y val
//    // (in the original code the calculation is
//    // x_val = x + x_offset/total_maxx - kernelRad - 1... not sure if I still need that)
//    float total_mass_inv = 1.f / total_mass;
//    float x_val = float (x) + x_offset * total_mass_inv - kernelRad - 1;
//    float y_val = float (y) + y_offset * total_mass_inv - kernelRad - 1;
//    float z_val = float (sliceIdx + 1);
//    float r2_val = r2_sum * total_mass_inv;
//
//    // Construct particle and return
//    Particle p{ x_val, y_val, z_val, total_mass, r2_val };
//    return p;
//}
//thrust_operator int SeverParticle::operator()(Particle& p)
//{
//    p.parent = nullptr;
//    p.nContributingSlices = 0;
//
//    return 0;
//}
//thrust_operator bool CheckParticleRadius::operator()(int idx)
//{
//    int ixPrev = idx % prevParticleCount;
//    int ixCur = idx / prevParticleCount;
//    Particle& prev = prevParticles[ixPrev];
//    Particle& cur = curParticles[ixCur];
//
//    float dX = prev.x - cur.x;
//    float dY = prev.y - cur.y;
//    float r2 = (dX * dX + dY * dY);
//    return r2 < r2Max;
//}
//thrust_operator int AttachParticle::operator() (const int idx)
//{
//    int ixPrev = idx % prevParticleCount;
//    int ixCur = idx / prevParticleCount;
//    Particle& prev = prevParticles[ixPrev];
//    Particle& cur = curParticles[ixCur];
//
//#if SOLVER_DEVICE
//    unsigned long long int* parentAddr = (unsigned long long int*) & cur.parent;
//    unsigned long long int newParent = (unsigned long long int)(prev.parent ? prev.parent : &prev);
//    unsigned long long int* matchAddr = (unsigned long long int*) & cur.match;
//    unsigned long long int newMatch = (unsigned long long int)(&prev);
//    atomicExch (parentAddr, newParent);
//    atomicExch (matchAddr, newMatch);
//    atomicExch (&cur.nContributingSlices, prev.nContributingSlices + 1);
//#else
//    Particle* newParent = prev.parent ? prev.parent : &prev;
//    cur.parent = prev.parent ? prev.parent : &prev;
//    cur.nContributingSlices = prev.nContributingSlices + 1;
//#endif
//
//    return 0;
//}
//thrust_operator bool ShouldSeverParticle::operator()(Particle p)
//{
//    return ((p.nContributingSlices >= minSlices) && (p.i > p.match->i)) || (p.nContributingSlices >= maxSlices);
//}