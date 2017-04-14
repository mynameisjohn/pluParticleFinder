PROJECT(pluParticleFinder)

# min cmake version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "-std=c++11")

# This must be set to Debug or Release (really just Debug or other...)
IF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  # debug define
  ADD_DEFINITIONS(-DDEBUG=1)
  # Add -g for unix
  IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  ENDIF(UNIX)
ELSE(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  # Add unix optimization flag
  IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2")
  ENDIF(UNIX)
  # I think this flag is cross platform
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -O2")
ENDIF(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

# Hard code CUDA to be on for now
# let CMake find CUDA
FIND_PACKAGE(CUDA QUIET REQUIRED)

# Set up sm arch (may want to change this)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch sm_50")

IF(WIN32)
  # OCV 3.1
  SET(OCV_INC_DIR $ENV{LIBDIR}/OpenCV/31/include)
  SET(OCV_LIB_DIR $ENV{LIBDIR}/OpenCV/31/lib/x64)

  # FreeImage (header is next to library)
  SET(FREEIMG_INC_DIR $ENV{LIBDIR}/FreeImage/lib/x64)
  SET(FREEIMG_LIB_DIR ${FREEIMG_INC_DIR})
  SET(FREEIMG_LIB FreeImage.lib)
ELSE(WIN32)
  # OCV, whatever they have in there
  SET(OCV_INC_DIR /usr/local/include)
  SET(OCV_LIB_DIR /usr/local/lib)

  # FreeImage (this is where apt put it for me)
  SET(FREEIMG_INC_DIR /usr/include)
  SET(FREEIMG_LIB_DIR /usr/lib/x86_64-linux-gnu)
  SET(FREEIMG_LIB freeimage)
ENDIF(WIN32)

# Set up link directories
LINK_DIRECTORIES(${OCV_LIB_DIR} ${FREEIMG_LIB_DIR})

# Include directories
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${OCV_INC_DIR}
    ${FREEIMG_INC_DIR})

# Source files, include files, scripts
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
file(GLOB KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu)

# Create source groups
SOURCE_GROUP("Kernels" FILES ${KERNELS})
SOURCE_GROUP("Source" FILES ${SOURCES})
SOURCE_GROUP("Include" FILES ${HEADERS})

# Add the pylGrooveMatrix executable, which depends on source, include, and scripts
ADD_EXECUTABLE(pluParticleFinder ${SOURCES} ${HEADERS} ${KERNELS})

# Link libraries with executable
TARGET_LINK_LIBRARIES(pluParticleFinder LINK_PUBLIC
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_highgui
  opencv_cudaimgproc
  opencv_cudaarithm
  opencv_cudafilters
  opencv_highgui
  opencv_core
  opencv_imgproc
  ${FREEIMG_LIB}
  ${CUDA_LIBRARIES})
