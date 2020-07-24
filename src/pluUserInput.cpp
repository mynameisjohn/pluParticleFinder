#if !defined (PLU_DLL)

#include "ParticleFinder.h"

#include <functional>
#include <opencv2/highgui.hpp>

template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
struct fun_ptr_helper
{
public:
    typedef std::function<_Res (_ArgTypes...)> function_type;

    static void bind (function_type&& f)
    {
        instance ().fn_.swap (f);
    }

    static void bind (const function_type& f)
    {
        instance ().fn_ = f;
    }

    static _Res invoke (_ArgTypes... args)
    {
        return instance ().fn_ (args...);
    }

    typedef decltype(&fun_ptr_helper::invoke) pointer_type;
    static pointer_type ptr ()
    {
        return &invoke;
    }

private:
    static fun_ptr_helper& instance ()
    {
        static fun_ptr_helper inst_;
        return inst_;
    }

    fun_ptr_helper () {}

    function_type fn_;
};


template <const size_t _UniqueId, typename _Res, typename... _ArgTypes>
typename fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::pointer_type
get_fn_ptr (const std::function<_Res (_ArgTypes...)>& f)
{
    fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::bind (f);
    return fun_ptr_helper<_UniqueId, _Res, _ArgTypes...>::ptr ();
}

template<typename T>
std::function<typename std::enable_if<std::is_function<T>::value, T>::type>
make_function (T* t)
{
    return { t };
}

void ParticleFinder::getUserInput (GpuMat d_Input)
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
        { gaussRadiusTBName, 6 },     // These are the
        { hwhmTBName, 4 },            // default values
        { dilationRadiusTBName, 3 },  // specified in the
        { particleThreshTBName, 5 }   // PLuTARC_testbed
    };

    const float trackBarResolution = 1000;
    for (auto& it : mapParamValues)
        it.second *= trackBarResolution;

    try
    {
        // Trackbar callback, implemented below
        std::function<void (int, void*)> trackBarCallback = [&](int pos, void* priv)
        {
            // Assign DSP params
            _gaussFiltRadius = mapParamValues[gaussRadiusTBName] / trackBarResolution;
            _HWHM = mapParamValues[hwhmTBName] / trackBarResolution;
            _dilationRadius = mapParamValues[dilationRadiusTBName] / trackBarResolution;
            _particleThreshold = mapParamValues[particleThreshTBName] / trackBarResolution;

            ResetKernels ();

            // TODO prevent bad values from existing

            GpuMat d_InputCirc = d_Input.clone ();

            // do DSP, get particles
            std::vector<FoundParticle> vParticlesInImg;
            doDSPAndFindParticlesInImg (0, 0, d_Input, &vParticlesInImg);

            // Draw circles in local max image
            if (!vParticlesInImg.empty ())
            {
                cv::Mat hImg;
                d_InputCirc.download (hImg);
                // This offset was needed for some reason
                for (FoundParticle& fp : vParticlesInImg)
                {
                    cv::circle (hImg, cv::Point (fp.fPosX, fp.fPosY), 5, cv::Scalar (0), -1);
                }
                d_InputCirc.upload (hImg);
            }

            // Show several of our images
            // returns formatted images for display
            auto makeDisplayImage = [](GpuMat& in)
            {
                GpuMat out;
                in.convertTo (out, CV_32F);
                RemapImage (out, 0, 1);
                return out;
            };

            // Create larger display image (4 images, corner to corner)
            cv::Size dataSize = d_Input.size ();
            cv::Size dispSize = dataSize;
            dispSize *= 2;    // Multiply by two in x and y
            GpuMat displayMat (dispSize, CV_32F, 0.f);

            // Display regions
            cv::Rect topLeft ({ 0, 0 }, dataSize);
            cv::Rect topRight (cv::Rect ({ dataSize.width, 0 }, dataSize));
            cv::Rect bottomLeft ({ 0, dataSize.height }, dataSize);
            cv::Rect bottomRight ({ dataSize.width, dataSize.height }, dataSize);

            // Copy all images to display image in correct place
            makeDisplayImage (d_Input).copyTo (displayMat (topLeft));
            makeDisplayImage (_procData[0].filteredImg).copyTo (displayMat (topRight));
            makeDisplayImage (_procData[0].dilatedImg).copyTo (displayMat (bottomLeft));
            makeDisplayImage (d_InputCirc).copyTo (displayMat (bottomRight));

            // Show new image
            cv::resizeWindow (windowName, dispSize.width, dispSize.height);
            cv::imshow (windowName, displayMat);
        };

        // Create window, just show input first
        cv::namedWindow (windowName, cv::WINDOW_OPENGL);

        // Create trackbars
        auto createTrackBar = [&mapParamValues, windowName, &trackBarCallback](std::string tbName, int maxVal)
        {
            auto it = mapParamValues.find (tbName);
            if (it != mapParamValues.end ())
            {
                cv::createTrackbar (tbName, windowName, &mapParamValues[tbName], maxVal, get_fn_ptr<0> (trackBarCallback));
            }
        };

        createTrackBar (gaussRadiusTBName, 15 * trackBarResolution);
        createTrackBar (hwhmTBName, 15 * trackBarResolution);
        createTrackBar (dilationRadiusTBName, 15 * trackBarResolution);
        createTrackBar (particleThreshTBName, 15 * trackBarResolution);

        // Call the callback on our own, just to pump things and show the images
        trackBarCallback (0, nullptr);

        // Wait while user sets things until they press a key (any key?)
        cv::waitKey ();

        // Destroy window
        cv::destroyWindow (windowName);
    }
    catch (cv::Exception e)
    {
        std::cout << e.what () << std::endl;
        std::cout << "Error creating user interface! Using default parameters\n" << std::endl;
    }
}

#endif // ! defined (PLU_DLL)