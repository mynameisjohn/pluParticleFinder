PROJECT(pluParticleFinder)

# min cmake version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "-std=c++11")

# default to debug
SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug or Release")

# configure debug / release flags
IF (CMAKE_BUILD_TYPE MATCHES Debug)
  # debug define
  ADD_DEFINITIONS(-DDEBUG=1)
  # Add -g for unix
  IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  ENDIF(UNIX)
ELSEIF (CMAKE_BUILD_TYPE MATCHES Release)
  # Add unix optimization flag
  IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2")
  ENDIF(UNIX)
  # I think this flag is cross platform
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -O2")
ENDIF (CMAKE_BUILD_TYPE MATCHES Debug)

SET(SOLVER_DEVICE 1 CACHE BOOL "Run the particle solver on the device")
SET(CUDA_ARCH "sm_50" CACHE STRING "CUDA Streaming Architecture version")

# Hard code CUDA to be on for now
# let CMake find CUDA
FIND_PACKAGE(CUDA QUIET REQUIRED)

SET(CUDA_ARCH "sm_50" CACHE STRING "CUDA Streaming Architecture version")

# Set up sm arch (may want to change this)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch ${CUDA_ARCH}")

SET(OCV_VERSION "412" CACHE STRING "OpenCV Version")

IF (CMAKE_BUILD_TYPE MATCHES Debug)
  SET(OCV_SUFFIX "${OCV_VERSION}d")
ELSEIF (CMAKE_BUILD_TYPE MATCHES Release)
  SET(OCV_SUFFIX "${OCV_VERSION}")
ENDIF (CMAKE_BUILD_TYPE MATCHES Debug)

IF(WIN32)
  IF(NOT OCV_INC_DIR)
    SET(OCV_INC_DIR $ENV{LIBDIR}/OpenCV/include CACHE STRING "OpenCV include directory")
  ENDIF(NOT OCV_INC_DIR)
  IF(NOT OCV_LIB_DIR)
    SET(OCV_LIB_DIR $ENV{LIBDIR}/OpenCV/lib/x64 CACHE STRING "OpenCV library directory")
  ENDIF(NOT OCV_LIB_DIR)

  # FreeImage (header is next to library)
  SET(FREEIMG_INC_DIR $ENV{LIBDIR}/FreeImage/lib/x64 CACHE STRING "FreeImage include directory")
  SET(FREEIMG_LIB_DIR ${FREEIMG_INC_DIR} CACHE STRING "FreeImage library directory directory")
  SET(FREEIMG_LIB FreeImage.lib)
  
ELSE(WIN32)
  IF(NOT OCV_INC_DIR)
    SET(OCV_INC_DIR /usr/local/include)
  ENDIF(NOT OCV_INC_DIR)
  IF(NOT OCV_LIB_DIR)
    SET(OCV_LIB_DIR /usr/local/lib)
  ENDIF(NOT OCV_LIB_DIR)

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
CUDA_ADD_EXECUTABLE(pluParticleFinder ${SOURCES} ${HEADERS} ${KERNELS})

# Link libraries with executable
TARGET_LINK_LIBRARIES(pluParticleFinder LINK_PUBLIC
  opencv_core${OCV_SUFFIX}
  opencv_imgproc${OCV_SUFFIX}
  opencv_imgcodecs${OCV_SUFFIX}
  opencv_highgui${OCV_SUFFIX}
  opencv_cudaimgproc${OCV_SUFFIX}
  opencv_cudaarithm${OCV_SUFFIX}
  opencv_cudafilters${OCV_SUFFIX}
  opencv_highgui${OCV_SUFFIX}
  opencv_core${OCV_SUFFIX}
  opencv_imgproc${OCV_SUFFIX}
  ${FREEIMG_LIB}
  ${CUDA_LIBRARIES})