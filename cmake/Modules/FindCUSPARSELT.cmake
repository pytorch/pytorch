# Find the CUSPARSELT library
#
# The following variables are optionally searched for defaults
#  CUSPARSELT_ROOT: Base directory where CUSPARSELT is found
#  CUSPARSELT_INCLUDE_DIR: Directory where CUSPARSELT header is searched for
#  CUSPARSELT_LIBRARY: Directory where CUSPARSELT library is searched for
#
# The following are set after configuration is done:
#  CUSPARSELT_FOUND
#  CUSPARSELT_INCLUDE_PATH
#  CUSPARSELT_LIBRARY_PATH

include(FindPackageHandleStandardArgs)

set(CUSPARSELT_ROOT $ENV{CUSPARSELT_ROOT_DIR} CACHE PATH "Folder containing NVIDIA cuSPARSELt")
if (DEFINED $ENV{CUSPARSELT_ROOT_DIR})
  message(WARNING "CUSPARSELT_ROOT_DIR is deprecated. Please set CUSPARSELT_ROOT instead.")
endif()
list(APPEND CUSPARSELT_ROOT $ENV{CUSPARSELT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# Compatible layer for CMake <3.12. CUSPARSELT_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUSPARSELT_ROOT})

set(CUSPARSELT_INCLUDE_DIR $ENV{CUSPARSELT_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA cuSPARSELt header files")

find_path(CUSPARSELT_INCLUDE_PATH cusparseLt.h
  HINTS ${CUSPARSELT_INCLUDE_DIR}
  PATH_SUFFIXES cuda/include cuda include)

set(CUSPARSELT_LIBRARY $ENV{CUSPARSELT_LIBRARY} CACHE PATH "Path to the cusparselt library file (e.g., libcusparseLt.so)")

set(CUSPARSELT_LIBRARY_NAME "libcusparseLt.so")
if(MSVC)
  set(CUSPARSELT_LIBRARY_NAME "cusparseLt.lib")
endif()

find_library(CUSPARSELT_LIBRARY_PATH ${CUSPARSELT_LIBRARY_NAME}
  PATHS ${CUSPARSELT_LIBRARY}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(CUSPARSELT DEFAULT_MSG CUSPARSELT_LIBRARY_PATH CUSPARSELT_INCLUDE_PATH)

if(CUSPARSELT_FOUND)
  # Get cuSPARSELt version
  file(READ ${CUSPARSELT_INCLUDE_PATH}/cusparseLt.h CUSPARSELT_HEADER_CONTENTS)
  string(REGEX MATCH "define CUSPARSELT_VER_MAJOR * +([0-9]+)"
               CUSPARSELT_VERSION_MAJOR "${CUSPARSELT_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUSPARSELT_VER_MAJOR * +([0-9]+)" "\\1"
               CUSPARSELT_VERSION_MAJOR "${CUSPARSELT_VERSION_MAJOR}")
  string(REGEX MATCH "define CUSPARSELT_VER_MINOR * +([0-9]+)"
               CUSPARSELT_VERSION_MINOR "${CUSPARSELT_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUSPARSELT_VER_MINOR * +([0-9]+)" "\\1"
               CUSPARSELT_VERSION_MINOR "${CUSPARSELT_VERSION_MINOR}")
  string(REGEX MATCH "define CUSPARSELT_VER_PATCH * +([0-9]+)"
               CUSPARSELT_VERSION_PATCH "${CUSPARSELT_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUSPARSELT_VER_PATCH * +([0-9]+)" "\\1"
               CUSPARSELT_VERSION_PATCH "${CUSPARSELT_VERSION_PATCH}")
  # Assemble cuSPARSELt version. Use minor version since current major version is 0.
  if(NOT CUSPARSELT_VERSION_MINOR)
    set(CUSPARSELT_VERSION "?")
  else()
    set(CUSPARSELT_VERSION
        "${CUSPARSELT_VERSION_MAJOR}.${CUSPARSELT_VERSION_MINOR}.${CUSPARSELT_VERSION_PATCH}")
  endif()
endif()

mark_as_advanced(CUSPARSELT_ROOT CUSPARSELT_INCLUDE_DIR CUSPARSELT_LIBRARY CUSPARSELT_VERSION)
