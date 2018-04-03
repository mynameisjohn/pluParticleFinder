using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class PLuTARC_Engine : MonoBehaviour {
    // Our DLL functions
    const string strPluginName = "pluParticleFinder";
    
    // We call this once in start to load the image files into
    // GPU memory - we return the total number of images (slices)
    [DllImport(strPluginName)]
    private static extern System.IntPtr CreateParticleFinder(string[] aImageFiles, int nImageFiles, int nStartOfStack, int nEndOfStack, ref int nSliceCount, ref int nPixelsX, ref int nPixelsY);
    
    [DllImport(strPluginName)]
    private static extern bool DestroyParticleFinder(System.IntPtr pParticleFinder);

    // Given current DSP parameters, this function kicks off a task
    // to find and link particles in the slices (cancelling any previous task)
    [DllImport(strPluginName)]
    private static extern bool StartFindingParticles(System.IntPtr pParticleFinder,
        int nGaussianRadius,
        float fHWHM,
        int nDilationRadius,
        float fParticleThreshold,
        int nMaskRadius,
        int nFeatureRadius,
        int nMinSliceCount,
        int nMaxSliceCount,
        int nNeighborRadius,
        int nMaxLevel);

    // We call this function to determine how many particles there
    // are in a given slice - returns negative if slice not yet ready
    [DllImport(strPluginName)]
    private static extern int GetParticleCountInSlice(System.IntPtr pParticleFinder, int ixSlice);

    [DllImport(strPluginName)]
    private static extern bool GetParticlesInSlice(System.IntPtr pParticleFinder, int ixSlice, ref System.IntPtr rpData, ref int rnParticles);

    [DllImport(strPluginName)]
    private static extern bool CancelParticleFindingTask(System.IntPtr pParticleFinder);

    // Public variables
    public string[] ImageFiles = new string[0];
    public int StartOfStack = 0;
    public int EndOfStack = 141;

    [Tooltip("Gaussian Kernel Radius")]
    [Range(3,9)]
    public int GaussianRadius = 6;
    [Tooltip("Gaussian HWHM")]
    [Range(2, 6)]
    public float HWHM = 4f;
    [Tooltip("Dilation Kernel Radius")]
    [Range(2, 6)]
    public int DilationRadius = 3;
    [Tooltip("Particle Threshold")]
    [Range(0.001f, 0.01f)]
    public float ParticleThreshold = .005f;

    [Tooltip("Mask Radius")]
    [Range(3, 11)]
    public int MaskRadius = 3;
    [Tooltip("Feature Radius")]
    [Range(3, 11)]
    public int FeatureRadius = 6;
    [Tooltip("Min Slices")]
    [Range(1, 5)]
    public int MinSliceCount = 3;
    [Tooltip("Max Slices")]
    [Range(3, 11)]
    public int MaxSliceCount = 5;
    [Tooltip("Neighbor Search Radius")]
    [Range(3, 11)]
    public int NeighborRadius = 6;
    [Tooltip("LOD downsampling level")]
    [Range(2, 5)]
    public int MaxLevel = 3;

    [Tooltip("Particle Scale")]
    [Range(1, 5)]
    public float ParticleScale = 1f;

    public bool Verbose = false;

    // Private variables
    System.IntPtr m_pParticleFinder = System.IntPtr.Zero;
    Coroutine m_coroFindParticles = null;
    bool m_bCoroRunning = false;
    bool m_bDirtyParamState = false;
    GameObject[] m_aSlices = null;

    Vector3 m_v3ParticlePlacementOfs = Vector3.zero;

    // Use this for initialization
    void Start()
    {
        // Initialize particle finder, which loads and uploads images from disk
        int nSliceCount = -1;
        string[] aImageFiles = new string[ImageFiles.Length];
        for (int i = 0; i < ImageFiles.Length; i++)
            aImageFiles[i] = System.IO.Path.Combine(Application.streamingAssetsPath, ImageFiles[i]);

        int nPixelsX = 0, nPixelsY = 0;
        m_pParticleFinder = CreateParticleFinder(aImageFiles, aImageFiles.Length, StartOfStack, EndOfStack, ref nSliceCount, ref nPixelsX, ref nPixelsY);
        if (m_pParticleFinder == System.IntPtr.Zero || nSliceCount < 0)
        {
            Debug.LogError("Unable to construct particle finder");
            Destroy(this);
        }
        else
        {
            // nSliceCount isn't supposed to change  so make room for particles
            m_aSlices = new GameObject[nSliceCount];

            m_v3ParticlePlacementOfs = -new Vector3(nPixelsX / 2f, nPixelsY / 2f, nSliceCount / 2f);
        }
    }

    void OnGUI()
    {
        // Flag dirty state, which kicks off particle finding
        if (m_coroFindParticles != null && m_bCoroRunning)
        {
            if (GUI.Button(new Rect(10, 10, 100, 30), "Stop"))
            {
                CancelParticleFindingTask(m_pParticleFinder);
                StopCoroutine(m_coroFindParticles);
                m_coroFindParticles = null;
                m_bCoroRunning = false;
            }
        }
        else
        {
            if (GUI.Button(new Rect(10, 10, 100, 30), "Start"))
            {
                SetDirty();
            }
/* Don't really need this, just use Unity UI
            // Gaussian Radius
            int nY = 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Gaussian Radius");
            GaussianRadius = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), GaussianRadius, 3, 9);
            // HWHM
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "HWHM");
            HWHM = GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), HWHM, 2, 6);
            // Dilatioin Radius
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Dilation Kernel Radius");
            DilationRadius = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), DilationRadius, 2 ,6);
            // Particle Threshold
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Particle Intensity Threshold");
            ParticleThreshold = GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), ParticleThreshold, 0.001f, 0.01f);
            // Particle Threshold
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Mask Radius");
            MaskRadius = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), MaskRadius, 3, 11);
            // Particle Threshold
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Feature Radius");
            FeatureRadius = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), FeatureRadius, 3, 11);
            // Min Slice Count
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Min Slice Count");
            MinSliceCount = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), MinSliceCount, 1, 5);
            // Min Slice Count
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Max Slice Count");
            MaxSliceCount = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), MaxSliceCount, 3, 11);
            // Min Slice Count
            nY += 40;
            GUI.Label(new Rect(10, nY, 100, 30), "Neighbor Radius");
            NeighborRadius = (int)GUI.HorizontalSlider(new Rect(10, nY + 20, 100, 30), NeighborRadius, 3, 11);
*/
        }
    }

    // This state gets picked up in update if it's dirty
    public void SetDirty()
    {
        m_bDirtyParamState = true;
    }

    // Make sure we clean up before we leave
    void OnDestroy()
    {
        StopAllCoroutines();
        DestroyParticleFinder(m_pParticleFinder);
        m_pParticleFinder = System.IntPtr.Zero;
        m_aSlices = null;
    }

    void DestroyExistingSlices()
    {
        for (int ixSlice = 0; ixSlice < m_aSlices.Length; ixSlice++)
        {
            if (m_aSlices[ixSlice])
            {
                Destroy(m_aSlices[ixSlice]);
                m_aSlices[ixSlice] = null;
            }
        }
    }
	
	// Update is called once per frame
	void Update ()
    {
        // Null out our coroutine if it's finished
		if (m_coroFindParticles != null && m_bCoroRunning == false)
        {
            m_coroFindParticles = null;
        }

        // See if we need to start it again
        if (m_bDirtyParamState)
        {
            // Clear flag
            m_bDirtyParamState = false;

            // Cancel any existing coroutines now
            if (m_coroFindParticles != null)
            {
                m_bCoroRunning = false;
                StopCoroutine(m_coroFindParticles);
            }

            // Start a new coroutine to find particles
            if (StartFindingParticles(m_pParticleFinder, GaussianRadius, HWHM, DilationRadius, ParticleThreshold, MaskRadius, FeatureRadius, MinSliceCount, MaxSliceCount, NeighborRadius, MaxLevel))
            {
                m_coroFindParticles = StartCoroutine(CoroGetSlice());
            }
        }
    }

    // We use this coroutine to check for new slice data by
    // looping over all our slices and waiting for a true
    // value to return from GetParticleCountInSlice, indicating
    // that the slice has been processed. If there are particles
    // in the slice, we create and merge particle geometry
    IEnumerator CoroGetSlice()
    {
        float tStart = Time.time;

        m_bCoroRunning = true;

        // Destroy existing slices
        DestroyExistingSlices();

        // Create particle geometry for each slice
        for (int ixSlice = 0; ixSlice < m_aSlices.Length; ixSlice++)
        {
            if (Verbose)
                Debug.Log("Finding particles in slice slice " + ixSlice);
            
            // Keep spinning until we get a valid slice count,
            // yield after each iteration to avoid stalling
            int nParticlesInSlice = -1;
            System.IntPtr pParticleData = System.IntPtr.Zero;
            while (GetParticlesInSlice(m_pParticleFinder, ixSlice, ref pParticleData, ref nParticlesInSlice) == false)
            {
                if (Verbose)
                    Debug.Log("Waiting for particles in slice " + ixSlice);
                yield return true;
            }

            // We have a particle count for this slice
            if (nParticlesInSlice > 0)
            {
                if (Verbose)
                    Debug.Log("Found " + nParticlesInSlice + " particles in slice " + ixSlice);

                // Use the float data to construct a vector of particle data
                // The x and y components are the xy position, and the third component
                // is the particle intensity (actual z pos is implied by slice index)
                int nFloatData = 4 * nParticlesInSlice;
                float[] afParticleData = new float[nFloatData];
                //yield return true;

                // I'm not sure if I can copy this directly...
                Marshal.Copy(pParticleData, afParticleData, 0, nFloatData);
                //yield return true;

                // Allocate mem for vector3s (could I copy to an array of vecs?)
                Vector3[] av3ParticleData = new Vector3[nParticlesInSlice];
                //yield return true;

                // Copy particle data into vector
                for (int ixParticle = 0; ixParticle < nParticlesInSlice; ixParticle++)
                {
                    int ixParticleData = 4 * ixParticle;
                    av3ParticleData[ixParticle] = new Vector3();
                    av3ParticleData[ixParticle].x = afParticleData[ixParticleData + 0];
                    av3ParticleData[ixParticle].y = afParticleData[ixParticleData + 1];
                    av3ParticleData[ixParticle].z = afParticleData[ixParticleData + 2];
                    //yield return true;
                }

                // Geometry stuff
                List<GameObject> liGameObjectsToDestroy = new List<GameObject>();
                List<CombineInstance> liCombineInstances = new List<CombineInstance>();
                List<List<CombineInstance>> liliCombineInstances = new List<List<CombineInstance>>();
                //yield return true;

                bool bFirst = true;
                Bounds SliceBounds = new Bounds();
                foreach (Vector3 v3ParticleData in av3ParticleData)
                {
                    // Create a sphere (TODO reflect scale with intensity or something)
                    GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    go.transform.localScale *= ParticleScale;
                    go.transform.position = new Vector3(v3ParticleData.x, v3ParticleData.y, (float)ixSlice) - m_v3ParticlePlacementOfs;
                    go.transform.SetParent(transform, false);
                    liGameObjectsToDestroy.Add(go);
                    //yield return true;

                    // Add to mesh combine
                    CombineInstance ci = new CombineInstance();
                    MeshFilter mf = go.GetComponent<MeshFilter>();
                    ci.mesh = mf.sharedMesh;
                    ci.transform = mf.transform.localToWorldMatrix;
                    go.SetActive(false);
                    //yield return true;

                    // See if we'll have too many verts, move to list if so
                    int nVertsPerSphere = mf.mesh.vertexCount;
                    int nVertsTotal = (1 + liCombineInstances.Count) * nVertsPerSphere;
                    // Debug.Log(nVertsPerSphere.ToString() + ", " + nVertsTotal.ToString());
                    if (nVertsTotal >= ushort.MaxValue)
                    {
                        // Debug.Log("Creating new mesh combine instance");
                        liliCombineInstances.Add(liCombineInstances);
                        liCombineInstances = new List<CombineInstance>();
                    }

                    liCombineInstances.Add(ci);
                    //yield return true;

                    if (bFirst)
                    {
                        bFirst = false;
                        SliceBounds = go.GetComponent<SphereCollider>().bounds;
                    }
                    else
                    {
                        SliceBounds.Encapsulate(go.GetComponent<Collider>().bounds);
                    }
                }

                // Scale slice bounds to particle scale - why doesn't this happen automatically
                SliceBounds.size = new Vector3(SliceBounds.size.x, SliceBounds.size.y, ParticleScale);

                // Any stragglers
                if (liCombineInstances.Count > 0)
                    liliCombineInstances.Add(liCombineInstances);
                liCombineInstances = new List<CombineInstance>(); // To deref, but why?
                //yield return true;

                // Create the slice owning all subslices
                // Debug.Log("We have " + liliCombineInstances.Count.ToString() + " subslices");
                GameObject goSlice = new GameObject();
                for (int i = 0; i < liliCombineInstances.Count; i++)
                {
                    // Each sub slice is a cube... but we get rid of the mesh?
                    GameObject goCombined = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    goCombined.name = "SubSlice_" + ixSlice.ToString() + "_" + i.ToString();
                    goCombined.GetComponent<MeshFilter>().mesh = new Mesh();
                    goCombined.GetComponent<MeshFilter>().mesh.CombineMeshes(liliCombineInstances[i].ToArray());
                    goCombined.transform.parent = goSlice.transform;
                    //yield return true;
                }
                // Add slice object to game and to our list
                goSlice.SetActive(true);
                goSlice.transform.parent = transform;
                goSlice.name = "Slice_" + ixSlice.ToString();
                goSlice.AddComponent<ParticleSlice>().SetBounds(SliceBounds);
                m_aSlices[ixSlice] = goSlice;
                //yield return true;

                // Destroy spheres we combined
                foreach (GameObject go in liGameObjectsToDestroy)
                {
                    Destroy(go);
                    //yield return true;
                }
            }

            yield return true;
        }

        float tEnd = Time.time;

        //if (Verbose)
        Debug.Log("Particle finding took " + (tEnd - tStart) + " seconds");

        // Get out
        m_bCoroRunning = false;
        yield break;
    }
}
