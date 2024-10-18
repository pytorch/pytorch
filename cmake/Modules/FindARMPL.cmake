# Build with ARM Performance Library backend for the Arm architecture
# Note: Performance Library is available from:
# https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries#Software-Download
#   and must be built separately. The location of the Performance Library build
#   must be set with the env var ARMPL_ROOT_DIR. This path will be checked later
#   as part of FindARMPL.cmake

# This module defines:
#  ARMPL_INCLUDE_DIRS, where to find the headers
#  ARMPL_LIBRARY, the libraries to link against

find_path(
  ARMPL_INCLUDE_DIR armpl.h
  PATHS
  $ENV{ARMPL_ROOT_DIR}/include
  /usr/local/include
  /usr/include
)

find_library(
  ARMPL_LIBRARY NAMES armpl_lp64
  PATHS
  $ENV{ARMPL_ROOT_DIR}/lib
  /usr/local/lib
  /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ARMPL DEFAULT_MSG ARMPL_INCLUDE_DIR ARMPL_LIBRARY)

if(ARMPL_FOUND)
  message(
    STATUS
    "Found ARMPL  (include: ${ARMPL_INCLUDE_DIR}, library: ${ARMPL_LIBRARY})")
  add_library(ARMPL::ARMPL UNKNOWN IMPORTED)
  set_property(
    TARGET ARMPL::ARMPL PROPERTY IMPORTED_LOCATION ${ARMPL_LIBRARY})
  set_property(
    TARGET ARMPL::ARMPL PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARMPL_INCLUDE_DIR})
  mark_as_advanced(ARMPL_INCLUDE_DIR ARMPL_LIBRARY)
endif()