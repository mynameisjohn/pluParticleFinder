#include "ParticleFinder.h"
#include <stdlib.h>

int main( int argc, char ** argv )
{
	ParticleFinder P;
	P.Execute( { 
		"phi41pct_3D_6zoom_0001.tif", 
		"phi41pct_3D_6zoom_0002.tif", 
		"phi41pct_3D_6zoom_0003.tif", }, 
		1, 141,
		true);
	return EXIT_SUCCESS;
}