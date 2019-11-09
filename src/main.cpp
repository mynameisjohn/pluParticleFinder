#if !defined (PLU_DLL)

#include "ParticleFinder.h"

#include <pugixml.hpp>

#include <stdlib.h>
#include <map>
#include <sstream>

int main( int argc, char ** argv )
{
    ParticleFinder P;
    std::list<std::string> inputFiles;
    int startOfStack{ 0 }, endOfStack{ 0 };
    bool getUserInput = true;

    if (argc == 2)
    {
        pugi::xml_document paramDoc;
        if (paramDoc.load_file (argv[1]))
        {
            auto params = paramDoc.child ("pluParams");
            if (!paramDoc.attribute ("user_input").empty ())
                getUserInput = paramDoc.attribute ("user_input").as_bool ();

            auto inputParams = params.child ("input");
            startOfStack = inputParams.attribute ("stack_start").as_int ();
            endOfStack = inputParams.attribute ("stack_end").as_int ();

            for (auto& inputFileEl : inputParams.children ())
                if (std::strcmp (inputFileEl.name (), "file") == 0)
                    inputFiles.push_back (inputFileEl.child_value ());

            auto cfParams = params.child ("centerfind_params");

            P.SetGaussianRadius (cfParams.attribute ("filter_radius").as_int ());
            P.SetDilationRadius (cfParams.attribute ("dilation_radius").as_int ());
            P.SetFWHM (cfParams.attribute ("HWHM").as_float ());
            P.SetParticleThreshold (cfParams.attribute ("particle_threshold").as_float ());

            P.GetSolver ()->SetFeatureRadius (cfParams.attribute ("feature_radius").as_int ());
            P.GetSolver ()->SetMaskRadius (cfParams.attribute ("mask_radius").as_int ());
        }
    }
    else if (argc == 5)
    {
        std::string infileStem = argv[1];
        startOfStack = std::atoi (argv[2]);
        endOfStack = std::atoi (argv[3]);
        int frameCount = std::atoi (argv[4]);

        for (int i = 0; i < frameCount; i++)
        {
            std::stringstream stacknumber_stream;
            stacknumber_stream.setf (std::ios::right, std::ios::adjustfield);
            stacknumber_stream.fill ('0');
            stacknumber_stream << std::setw (4) << i;
            inputFiles.push_back (infileStem + "_" + stacknumber_stream.str () + ".tif");
        }

        getUserInput = true;
    }
    else 
    {
        std::cout << "Invalid params" << std::endl;
        return EXIT_FAILURE;
    }

    P.Initialize (inputFiles, startOfStack, endOfStack, getUserInput);

    P.Execute ();

    return EXIT_SUCCESS;
}
#endif // ! defined (PLU_DLL)