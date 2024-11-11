# This will define the following variables:
# SYCL_FOUND               : True if the system has the SYCL library.
# SYCL_INCLUDE_DIR         : Include directories needed to use SYCL.
# SYCL_LIBRARY_DIR         ：The path to the SYCL library.
# SYCL_LIBRARY             : SYCL library fullname.

include(FindPackageHandleStandardArgs)

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
endif()

string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
if(nosyclfound)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Find include path from binary.
find_file(
  SYCL_INCLUDE_DIR
  NAMES include
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# Find include/sycl path from include path.
find_file(
  SYCL_INCLUDE_SYCL_DIR
  NAMES sycl
  HINTS ${SYCL_ROOT}/include/
  NO_DEFAULT_PATH
  )

# Due to the unrecognized compilation option `-fsycl` in other compiler.
list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

# Find library directory from binary.
find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# Find SYCL library fullname.
# Don't use if(LINUX) here since this requires cmake>=3.25 and file is installed
# and used by other projects.
# See: https://cmake.org/cmake/help/v3.25/variable/LINUX.html
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  find_library(
    SYCL_LIBRARY
    NAMES sycl-preview
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
endif()
# On Windows, currently there's no sycl.lib. Only sycl7.lib with version suffix,
# where the current version of the SYCL runtime is 7.
# Until oneAPI adds support to sycl.lib without the version suffix,
# sycl_runtime_version needs to be hardcoded and uplifted when SYCL runtime version uplifts.
# TODO: remove this when sycl.lib is supported on Windows
if(WIN32)
  set(sycl_runtime_version 7)
  find_library(
    SYCL_LIBRARY
    NAMES "sycl${sycl_runtime_version}"
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  if(SYCL_LIBRARY STREQUAL "SYCL_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Cannot find a SYCL library on Windows")
  endif()
endif()

find_library(
  OCL_LIBRARY
  NAMES OpenCL
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT SYCL_INCLUDE_DIR) OR (NOT SYCL_LIBRARY_DIR) OR (NOT SYCL_LIBRARY))
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}")
