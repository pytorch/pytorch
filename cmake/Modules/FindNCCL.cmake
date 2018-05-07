# Try to find NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(NCCL_ROOT_DIR $ENV{NCCL_ROOT_DIR} CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
  ${NCCL_INCLUDE_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include)

if ($ENV{USE_STATIC_NCCL})
  message(STATUS "USE_STATIC_NCCL detected. Linking against static NCCL library")
  set(NCCL_LIBNAME "libnccl_static.a")
else()
  set(NCCL_LIBNAME "nccl")
endif()

find_library(NCCL_LIBRARY
  NAMES ${NCCL_LIBNAME}
  HINTS
  ${NCCL_LIB_DIR}
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/lib
  ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NCCL_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if (NCCL_FOUND)
  set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/nccl.h")
  message(STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
  file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
        REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
  if (NCCL_MAJOR_VERSION_DEFINED)
    string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
            NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
    message(STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}")
  endif()
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()
