#include "ParticleFinder.h"
#include "ParticleSolver.h"

ParticleFinder::Solver::Solver():
    _solverImpl( new impl() )
{}

ParticleFinder::Solver::~Solver () {}

void ParticleFinder::Solver::Init (int N)
{
    return _solverImpl->Init (N);
}

void ParticleFinder::Solver::ResetLinking ()
{
    _solverImpl->_stackToParticles.clear ();
}

int ParticleFinder::Solver::FindParticlesInImage(int stackNum, 
                                 int sliceIdx,
                                 GpuMat input, 
                                 GpuMat filteredImage, 
                                 GpuMat threshImg, 
                                 GpuMat particleImg,
                                 std::vector<FoundParticle> * particlesInImg /*= nullptr*/ )
{
    return _solverImpl->FindParticlesInImage( stackNum, sliceIdx, input, filteredImage, threshImg, particleImg, particlesInImg);
}

std::map<int, std::vector<ParticleFinder::FoundParticle>> ParticleFinder::Solver::LinkFoundParticles ()
{
    return _solverImpl->LinkFoundParticles ();
}

void ParticleFinder::Solver::SetMaskRadius( int mR )
{
    _solverImpl->_maskRadius = mR;
}

int ParticleFinder::Solver::GetMaskRadius () const
{
    return _solverImpl->_maskRadius;
}

void ParticleFinder::Solver::SetFeatureRadius( int fR )
{
    _solverImpl->_featureRadius = fR;
}

int ParticleFinder::Solver::GetFeatureRadius () const
{
    return _solverImpl->_featureRadius;
}

void ParticleFinder::Solver::SetMinSliceCount( int nMinSC )
{
    _solverImpl->_minSliceCount = nMinSC;
}

int ParticleFinder::Solver::GetMinSliceCount () const
{
    return _solverImpl->_minSliceCount;
}

void ParticleFinder::Solver::SetMaxSliceCount( int nMaxSC )
{
    _solverImpl->_maxSliceCount = nMaxSC;
}

int ParticleFinder::Solver::GetMaxSliceCount() const
{
    return _solverImpl->_maxSliceCount;
}

void ParticleFinder::Solver::SetNeighborRadius (float nR)
{
    _solverImpl->_neighborRadius = nR;
}

float ParticleFinder::Solver::GetNeighborRadius() const
{
    return _solverImpl->_neighborRadius;
}

void ParticleFinder::Solver::SetXYFactor (float xFactor)
{
    _solverImpl->_xyFactor = xFactor;
}

float ParticleFinder::Solver::GetXYFactor () const
{
    return _solverImpl->_xyFactor;
}

void ParticleFinder::Solver::SetZFactor (float zFactor)
{
    _solverImpl->_zFactor = zFactor;
}

float ParticleFinder::Solver::GetZFactor () const
{
    return _solverImpl->_zFactor;
}