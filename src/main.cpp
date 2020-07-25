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
    bool getUserInput = false;
    bool linkParticles = false;
    pugi::xml_document paramDoc;
    std::string outputFile2D;

    if (argc == 2 && paramDoc.load_file (argv[1]))
    {
        auto params = paramDoc.child ("pluParams");

        if (!params.attribute ("user_input").empty ())
            getUserInput = params.attribute ("user_input").as_bool ();

        if (!params.attribute ("link_particles").empty ())
            linkParticles = params.attribute ("link_particles").as_bool ();

        if (params.attribute("outputfile_2D"))
            outputFile2D = params.attribute ("outputfile_2D").as_string ();

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

        if (linkParticles)
        {
            auto linkParams = params.child ("link_params");
            if (linkParams.attribute ("outputfile_XYZT").as_string ())
                P.SetOutputFileXYZT (linkParams.attribute ("outputfile_XYZT").as_string ());
            P.GetSolver ()->SetFeatureRadius (linkParams.attribute ("feature_radius").as_int ());
            P.GetSolver ()->SetMaskRadius (linkParams.attribute ("mask_radius").as_int ());
            P.GetSolver ()->SetNeighborRadius (linkParams.attribute ("neighbor_radius").as_int ());
            P.GetSolver ()->SetMinSliceCount (linkParams.attribute ("min_slices").as_int ());
            P.GetSolver ()->SetMaxSliceCount (linkParams.attribute ("max_slices").as_int ());
            P.GetSolver ()->SetXYFactor (linkParams.attribute ("xy_factor").as_float ());
            P.GetSolver ()->SetZFactor (linkParams.attribute ("z_factor").as_float ());
        }
    }
    else if (argc == 6)
    {
        std::string infileStem = argv[1];
        outputFile2D = argv[2];
        startOfStack = std::atoi (argv[3]);
        endOfStack = std::atoi (argv[4]);
        int stackCount = std::atoi (argv[5]);

        for (int i = 0; i < stackCount; i++)
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
    
    P.SetOutputFile2D (outputFile2D);

    P.Execute (linkParticles);

    return EXIT_SUCCESS;
}
#endif // ! defined (PLU_DLL)