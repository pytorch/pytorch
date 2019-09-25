# - Config file for the Gloo package
# It defines the following variables
#  GLOO_INCLUDE_DIRS       - include directories for Gloo
#  GLOO_LIBRARIES          - libraries to link against
#  GLOO_CUDA_LIBRARIES     - cuda libraries to link against
#  GLOO_HIP_LIBRARIES      - hip libraries to link against

# library version information

set(GLOO_VERSION_MAJOR 0)
set(GLOO_VERSION_MINOR 5)
set(GLOO_VERSION_PATCH 0)
set(GLOO_VERSION "0.5.0")

# import targets
include ("${CMAKE_CURRENT_LIST_DIR}/GlooTargets.cmake")

# include directory and libraries.
#
# Newer versions of CMake set the INTERFACE_INCLUDE_DIRECTORIES property
# of the imported targets. It is hence not necessary to add this path
# manually to the include search path for targets which link to gflags.
# The following lines are here for backward compatibility, in case one
# would like to use the old-style target names.

get_filename_component(
    CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
# Note: the current list dir is _INSTALL_PREFIX/share/cmake/Gloo.
get_filename_component(
    _INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
set(GLOO_INCLUDE_DIRS "${_INSTALL_PREFIX}/include")

set(GLOO_LIBRARIES gloo)
set(GLOO_CUDA_LIBRARIES gloo_cuda gloo)
set(GLOO_HIP_LIBRARIES gloo_hip gloo)
