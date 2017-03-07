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

include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")

find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

find_library(CUDNN_LIBRARY cudnn
    PATHS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

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
    
