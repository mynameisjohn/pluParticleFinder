using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public class ParticleSlice : MonoBehaviour
{
    public Camera cam = null;
    private void Start()
    {
        if (cam == null)
        {
            GameObject[] cams = GameObject.FindGameObjectsWithTag("MainCamera");
            if (cams.Length ==0)
            {
                Destroy(this);
            }
            cam = cams[0].GetComponent<Camera>();
        }
    }
    public void SetBounds(Bounds b)
    {
        GetComponent<BoxCollider>().center = b.center;
        GetComponent<BoxCollider>().size = b.size;
    }
    // Update is called once per frame
    bool bOver = false;
	void Update ()
    {
        RaycastHit hit;
        Ray ray = cam.ScreenPointToRay(Input.mousePosition);

        if (Physics.Raycast(ray, out hit))
        {
            if (hit.transform == transform)
            {
                bOver = true;
                foreach (MeshRenderer r in GetComponentsInChildren<MeshRenderer>())
                    r.material.color = Color.red;
            }
            else if (bOver)
            {
                bOver = false;
                foreach (MeshRenderer r in GetComponentsInChildren<MeshRenderer>())
                    r.material.color = Color.white;
            }
        }
        else if (bOver)
        {
            bOver = false;
            foreach (MeshRenderer r in GetComponentsInChildren<MeshRenderer>())
                r.material.color = Color.white;
        }
    }
}
