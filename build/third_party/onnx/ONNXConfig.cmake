# - Config file for the ONNX package
# It defines the following variable(s)
#   ONNX_INCLUDE_DIRS     - include directories for FooBar
# as well as ONNX targets for other cmake libraries to use.

# library version information
set(ONNX_VERSION "1.5.0")

# import targets
include ("${CMAKE_CURRENT_LIST_DIR}/ONNXTargets.cmake")

# include directory.
#
# Newer versions of CMake set the INTERFACE_INCLUDE_DIRECTORIES property
# of the imported targets. It is hence not necessary to add this path
# manually to the include search path for targets which link to gflags.
# The following lines are here for backward compatibility, in case one
# would like to use the old-style include path.
get_filename_component(
    CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(
    _INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
set(ONNX_INCLUDE_DIRS "${_INSTALL_PREFIX}/include")
