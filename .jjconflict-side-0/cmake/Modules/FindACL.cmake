# Copied from: https://github.com/oneapi-src/oneDNN/blob/main/cmake/FindACL.cmake
# ----------
# FindACL
# ----------
#
# Finds the Arm Compute Library
# https://arm-software.github.io/ComputeLibrary/latest/
#
# This module defines the following variables:
#
#   ACL_FOUND          - True if ACL was found
#   ACL_INCLUDE_DIRS   - include directories for ACL
#   ACL_LIBRARIES      - link against this library to use ACL
#
# The module will also define two cache variables:
#
#   ACL_INCLUDE_DIR    - the ACL include directory
#   ACL_LIBRARY        - the path to the ACL library
#

# Use ACL_ROOT_DIR environment variable to find the library and headers
find_path(ACL_INCLUDE_DIR
  NAMES arm_compute/graph.h
  PATHS ENV ACL_ROOT_DIR
  )

find_library(ACL_LIBRARY
  NAMES arm_compute
  PATHS ENV ACL_ROOT_DIR
  PATH_SUFFIXES lib build
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACL DEFAULT_MSG
  ACL_INCLUDE_DIR
  ACL_LIBRARY
)

mark_as_advanced(
  ACL_LIBRARY
  ACL_INCLUDE_DIR
  )

# Find the extra libraries and include dirs
if(ACL_FOUND)
  find_path(ACL_EXTRA_INCLUDE_DIR
    NAMES half/half.hpp
    PATHS ENV ACL_ROOT_DIR
    PATH_SUFFIXES include
    )

  find_library(ACL_GRAPH_LIBRARY
    NAMES arm_compute_graph
    PATHS ENV ACL_ROOT_DIR
    PATH_SUFFIXES lib build
    )

  list(APPEND ACL_INCLUDE_DIRS
    ${ACL_INCLUDE_DIR} ${ACL_EXTRA_INCLUDE_DIR})
  list(APPEND ACL_LIBRARIES
    ${ACL_LIBRARY} ${ACL_GRAPH_LIBRARY})
endif()
