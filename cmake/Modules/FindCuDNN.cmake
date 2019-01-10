# - Try to find cuDNN
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT_DIR:            Base directory where all cuDNN components are found
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_DIRS
#  CUDNN_LIBRARIES
#  CUDNN_LIBRARY_DIRS
#
# Borrowed from https://github.com/caffe2/caffe2/blob/master/cmake/Modules/FindCuDNN.cmake

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

if($ENV{CUDNN_INCLUDE_DIR})
  SET(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR})
else($ENV{CUDNN_INCLUDE_DIR})
  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
endif($ENV{CUDNN_INCLUDE_DIR})

IF ($ENV{USE_STATIC_CUDNN})
  MESSAGE(STATUS "USE_STATIC_CUDNN detected. Linking against static CUDNN library")
  SET(CUDNN_LIBNAME "libcudnn_static.a")
ELSE()
  SET(CUDNN_LIBNAME "cudnn")
ENDIF()

if($ENV{CUDNN_LIBRARY})
  SET(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY})
else($ENV{CUDNN_LIBRARY})
  find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
    HINTS ${CUDNN_LIB_DIR} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
endif($ENV{CUDNN_LIBRARY})

find_package_handle_standard_args(
    CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(CUDNN_FOUND)
	# get cuDNN version
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
	string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
				 CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
				 CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
	string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
				 CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
				 CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
	string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
				 CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
				 CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()

  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_DIR}, library: ${CUDNN_LIBRARY})")
  mark_as_advanced(CUDNN_ROOT_DIR CUDNN_LIBRARY CUDNN_INCLUDE_DIR)
endif()
