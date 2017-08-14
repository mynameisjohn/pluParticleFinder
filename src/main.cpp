#include "ParticleFinder.h"
#include <stdlib.h>
#include <map>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport) 
#else
#define DLL_EXPORT
#endif

// Mockup of various C# interfaces

// We call this once in start to load the image files into
// GPU memory - we return the total number of images (slices)
extern "C" 
{
	// Given current DSP parameters, this function kicks off a task
	// to find and link particles in the slices (cancelling any previous task)
	//using ParticleFindingTask = std::pair<std::shared_ptr<std::atomic_bool>, std::future<std::vector<ParticleFinder::FoundParticle>>>;
	static std::map<ParticleFinder *, std::shared_ptr<ParticleFinder::AsyncParticleFindingTask>> s_mapParticleFindingTasks;
	DLL_EXPORT bool StartFindingParticles(ParticleFinder * pParticleFinder,
		int nGaussianRadius,
		float fHWHM,
		int nDilationRadius,
		float fParticleThreshold,
		int nMaskRadius,
		int nFeatureRadius,
		int nMinSliceCount,
		int nMaxSliceCount,
		int nNeighborRadius,
		int nMaxLevel)
	{
		if (pParticleFinder) 
		{
			// Make shared pointer to task (why not?)
			std::shared_ptr<ParticleFinder::AsyncParticleFindingTask> spParticleFindingTask;
			spParticleFindingTask = std::make_shared<ParticleFinder::AsyncParticleFindingTask>();

			// Fill in params
			spParticleFindingTask->pParticleFinder = pParticleFinder;
			spParticleFindingTask->nGaussianRadius = nGaussianRadius;
			spParticleFindingTask->fHWHM = fHWHM;
			spParticleFindingTask->nDilationRadius = nDilationRadius;
			spParticleFindingTask->fParticleThreshold = fParticleThreshold;
			spParticleFindingTask->nMaskRadius = nMaskRadius;
			spParticleFindingTask->nFeatureRadius = nFeatureRadius;
			spParticleFindingTask->nMinSliceCount = nMinSliceCount;
			spParticleFindingTask->nMaxSliceCount = nMaxSliceCount;
			spParticleFindingTask->nNeighborRadius = nNeighborRadius;
			spParticleFindingTask->nMaxLevel = nMaxLevel;

			// Cancel any existing task
			auto it = s_mapParticleFindingTasks.find(pParticleFinder);
			if (it != s_mapParticleFindingTasks.end())
			{
				if (it->second)
				{
					LockMutex(it->second->muData);
					it->second->bCancel = true;
				}
				s_mapParticleFindingTasks.erase(it);
			}

			// Kick off our task, it can be cancelled above later
			pParticleFinder->Execute(spParticleFindingTask);
			s_mapParticleFindingTasks[pParticleFinder] = spParticleFindingTask;

			return true;
		}
		return false;
	}

	// We call this function to determine how many particles there
	// are in a given slice - returns negative if slice not yet ready
	DLL_EXPORT bool GetParticlesInSlice(ParticleFinder * pParticleFinder, int ixSlice, float ** ppData, int * pnParticles)
	{
		// See if task is done
		std::vector<ParticleFinder::FoundParticle> * pvFoundParticles = nullptr;
		auto it = s_mapParticleFindingTasks.find(pParticleFinder);
		if (it != s_mapParticleFindingTasks.end())
		{
			// A vector is added to the task's map once its been processed
			LockMutex(it->second->muData);
			if (it->second->mapFoundSliceToParticles.count(ixSlice))
			{
				pvFoundParticles = &it->second->mapFoundSliceToParticles[ixSlice];
			}
		}

		// If we had a vector, assign the pointer to its data
		if (pvFoundParticles)
		{
			if (pnParticles)
				*pnParticles = (int)pvFoundParticles->size();
			if (ppData)
				*ppData = (float *)pvFoundParticles->data();

			return true;
		}

		// We haven't found particles in it yet
		return false;
	}

	DLL_EXPORT ParticleFinder * CreateParticleFinder(const char ** aszImageFiles, int nImageFiles, int nStartOfStack, int nEndOfStack, int * pnSliceCount)
	{
		if (ParticleFinder * pParticleFinder = new ParticleFinder())
		{
			if (pParticleFinder->Initialize(std::list<std::string>(aszImageFiles, aszImageFiles + nImageFiles), nStartOfStack, nEndOfStack))
			{
				if (pnSliceCount)
					*pnSliceCount = pParticleFinder->GetNumImages();
				return pParticleFinder;
			}
			delete pParticleFinder;
		}

		return nullptr;
}

	DLL_EXPORT bool DestroyParticleFinder(ParticleFinder * pParticleFinder)
	{
		// Cancel any existing task
		auto it = s_mapParticleFindingTasks.find(pParticleFinder);
		if (it != s_mapParticleFindingTasks.end())
		{
			if (it->second)
			{
				LockMutex(it->second->muData);
				it->second->bCancel = true;
			}
			s_mapParticleFindingTasks.erase(it);
		}

		if (pParticleFinder)
			delete pParticleFinder;
		return (bool)pParticleFinder;
	}

}

#ifndef PLU_DLL
int main( int argc, char ** argv )
{
	ParticleFinder P;
	P.Initialize({
		"phi41pct_3D_6zoom_0001.tif",
		"phi41pct_3D_6zoom_0002.tif",
		"phi41pct_3D_6zoom_0003.tif", },
		1, 141, true);
	P.Execute();
	return EXIT_SUCCESS;
}
#endif