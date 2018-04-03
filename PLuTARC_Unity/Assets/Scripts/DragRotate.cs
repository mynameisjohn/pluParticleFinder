using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DragRotate : MonoBehaviour
{
    private float _sensitivity;
    private Vector3 _mouseReference;
    private Vector3 _mouseOffset;
    private Vector3 _rotation;
    private bool _isRotating;
    public Camera cam = null;
    private void Start()
    {
        if (cam == null)
        {
            GameObject[] cams = GameObject.FindGameObjectsWithTag("MainCamera");
            if (cams.Length == 0)
            {
                Destroy(this);
            }
            cam = cams[0].GetComponent<Camera>();
        }
        _sensitivity = 0.4f;
        _rotation = Vector3.zero;
    }

    void Update()
    {
        if (!_isRotating && Input.GetMouseButton(0))
        {
            _isRotating = true;
            _mouseReference = Input.mousePosition;
        }
        else if (_isRotating && !Input.GetMouseButton(0))
        {
            _isRotating = false;
        }
        if (_isRotating)
        {
            // offset
            _mouseOffset = _sensitivity * (Input.mousePosition - _mouseReference);
            transform.RotateAround(transform.position, new Vector3(0, 1, 0), -_mouseOffset.x);
            transform.RotateAround(transform.position, new Vector3(1, 0, 0), -_mouseOffset.y);
            // _mouseOffset = cam.ScreenToWorldPoint(Input.mousePosition) - cam.ScreenToWorldPoint(_mouseReference);

            // apply rotation
            //_rotation.y = -(_mouseOffset.x + _mouseOffset.y) * _sensitivity;

            //// rotate
            //transform.Rotate(_rotation);

            // store mouse
            _mouseReference = Input.mousePosition;
        }
    }

    void OnMouseDown()
    {
        // rotating flag
        _isRotating = true;

        // store mouse
        _mouseReference = Input.mousePosition;
    }

    void OnMouseUp()
    {
        // rotating flag
        _isRotating = false;
    }
}