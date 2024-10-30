# This will define the following variables:
# SYCL_FOUND               : True if the system has the SYCL library.
# SYCL_INCLUDE_DIR         : Include directories needed to use SYCL.
# SYCL_LIBRARY_DIR         ：The path to the SYCL library.
# SYCL_LIBRARY             : SYCL library fullname.
# SYCL_COMPILER_VERSION    : SYCL compiler version.

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

# Find include/sycl/version.hpp to fetch sycl compiler version
find_file(
  SYCL_VERSION_HEADER_FILE
  NAMES version.hpp
  HINTS ${SYCL_INCLUDE_SYCL_DIR}
  NO_DEFAULT_PATH
  )

if(NOT SYCL_VERSION_HEADER_FILE)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "Cannot find include/sycl/version.hpp to get SYCL_COMPILER_VERSION!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Read the sycl version header file into a variable
file(READ ${SYCL_VERSION_HEADER_FILE} SYCL_VERSION_HEADER_CONTENT)

# Extract the SYCL compiler version from the version header content.
# 1. Match the regular expression to find the version string.
# 2. Replace the "__SYCL_COMPILER_VERSION" part with an empty string.
# 3. Strip leading and trailing spaces to get the version number.
string(REGEX MATCH "__SYCL_COMPILER_VERSION[ ]+[0-9]+" _SYCL_COMPILER_VERSION_MATCH ${SYCL_VERSION_HEADER_CONTENT})
string(REPLACE "__SYCL_COMPILER_VERSION" "" _SYCL_COMPILER_VERSION_NUMBER ${_SYCL_COMPILER_VERSION_MATCH})
string(STRIP ${_SYCL_COMPILER_VERSION_NUMBER} SYCL_COMPILER_VERSION)

# Find library directory from binary.
find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

# By default, we use libsycl.so on Linux and sycl.lib on Windows as the SYCL library name.
if (SYCL_COMPILER_VERSION VERSION_LESS_EQUAL 20240703)
  # Don't use if(LINUX) here since this requires cmake>=3.25 and file is installed
  # and used by other projects.
  # See: https://cmake.org/cmake/help/v3.25/variable/LINUX.html
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(sycl_lib_suffix "-preview")
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # On Windows, the SYCL library is named sycl7.lib until compiler version 20240703.
    # sycl.lib is supported in the later version.
    set(sycl_lib_sufix "7")
  endif()
endif()
 
# Find SYCL library fullname.
find_library(
  SYCL_LIBRARY
  NAMES "sycl${sycl_lib_suffix}"
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

# Find OpenCL library fullname, which is a dependency of oneDNN.
find_library(
  OCL_LIBRARY
  NAMES OpenCL
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT SYCL_LIBRARY) OR (NOT OCL_LIBRARY))
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL library is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

find_package_handle_standard_args(
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
  VERSION_VAR SYCL_COMPILER_VERSION
  )
