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
find_library(
  SYCL_LIBRARY
  NAMES sycl
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

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
