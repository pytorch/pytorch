# Find the CUDSS library
#
# The following variables are optionally searched for defaults
#  CUDSS_ROOT: Base directory where CUDSS is found
#  CUDSS_INCLUDE_DIR: Directory where CUDSS header is searched for
#  CUDSS_LIBRARY: Directory where CUDSS library is searched for
#
# The following are set after configuration is done:
#  CUDSS_FOUND
#  CUDSS_INCLUDE_PATH
#  CUDSS_LIBRARY_PATH

include(FindPackageHandleStandardArgs)

set(CUDSS_ROOT $ENV{CUDSS_ROOT_DIR} CACHE PATH "Folder containing NVIDIA CUDSS")
if (DEFINED $ENV{CUDSS_ROOT_DIR})
  message(WARNING "CUDSS_ROOT_DIR is deprecated. Please set CUDSS_ROOT instead.")
endif()
list(APPEND CUDSS_ROOT $ENV{CUDSS_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# Compatible layer for CMake <3.12. CUDSS_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUDSS_ROOT})

set(CUDSS_INCLUDE_DIR $ENV{CUDSS_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA CUDSS header files")

find_path(CUDSS_INCLUDE_PATH cudss.h
  HINTS ${CUDSS_INCLUDE_DIR}
  PATH_SUFFIXES cuda/include cuda include)

set(CUDSS_LIBRARY $ENV{CUDSS_LIBRARY} CACHE PATH "Path to the CUDSS library file (e.g., libcudss.so)")

set(CUDSS_LIBRARY_NAME "libcudss.so")
if(MSVC)
  set(CUDSS_LIBRARY_NAME "cudss.lib")
endif()

find_library(CUDSS_LIBRARY_PATH ${CUDSS_LIBRARY_NAME}
  PATHS ${CUDSS_LIBRARY}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(CUDSS DEFAULT_MSG CUDSS_LIBRARY_PATH CUDSS_INCLUDE_PATH)

if(CUDSS_FOUND)
  # Get CUDSS version
  file(READ ${CUDSS_INCLUDE_PATH}/cudss.h CUDSS_HEADER_CONTENTS)
  string(REGEX MATCH "define CUDSS_VER_MAJOR * +([0-9]+)"
               CUDSS_VERSION_MAJOR "${CUDSS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDSS_VER_MAJOR * +([0-9]+)" "\\1"
               CUDSS_VERSION_MAJOR "${CUDSS_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDSS_VER_MINOR * +([0-9]+)"
               CUDSS_VERSION_MINOR "${CUDSS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDSS_VER_MINOR * +([0-9]+)" "\\1"
               CUDSS_VERSION_MINOR "${CUDSS_VERSION_MINOR}")
  string(REGEX MATCH "define CUDSS_VER_PATCH * +([0-9]+)"
               CUDSS_VERSION_PATCH "${CUDSS_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDSS_VER_PATCH * +([0-9]+)" "\\1"
               CUDSS_VERSION_PATCH "${CUDSS_VERSION_PATCH}")
  # Assemble CUDSS version. Use minor version since current major version is 0.
  if(NOT CUDSS_VERSION_MINOR)
    set(CUDSS_VERSION "?")
  else()
    set(CUDSS_VERSION
        "${CUDSS_VERSION_MAJOR}.${CUDSS_VERSION_MINOR}.${CUDSS_VERSION_PATCH}")
  endif()
endif()

mark_as_advanced(CUDSS_ROOT CUDSS_INCLUDE_DIR CUDSS_LIBRARY CUDSS_VERSION)
