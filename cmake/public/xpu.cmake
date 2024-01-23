# XPU backend stacks (compiler, runtime, libraries, tools)

# output variables:
#
#  PYTORCH_FOUND_XPU
#  -- The flag to indicate whether XPU backend stacks are setup successfully or not.
#
#  PYTORCH_SYCL_LIBRARIES
#  -- Full name of SYCL runtime libraries.

# output INTERFACE target:
#
#  torch::syclrt
#  -- Define it for linking usage, e.g. target_link_library(tgt torch::syclrt)
#

set(PYTORCH_FOUND_XPU FALSE)

if(SYCL_cmake_included)
  return()
endif()
set(SYCL_cmake_included true)

# SYCL compiler and runtime setup
find_package(SYCLToolkit REQUIRED)
if(NOT SYCLTOOLKIT_FOUND)
  message("Cannot find SYCL compiler tool kit!")
  return()
endif()

# Try to find SYCL compiler version.hpp header
find_file(SYCL_VERSION
    NAMES version.hpp
    PATHS
        ${SYCL_INCLUDE_DIR}
    PATH_SUFFIXES
        sycl
        sycl/CL
        sycl/CL/sycl
    NO_DEFAULT_PATH)

if(NOT SYCL_VERSION)
  message("Can NOT find SYCL version file!")
  return()
endif()

find_library(PYTORCH_SYCL_LIBRARIES sycl HINTS ${SYCL_LIBRARY_DIR})

# SYCL runtime cmake target
add_library(torch::syclrt INTERFACE IMPORTED)
set_property(TARGET torch::syclrt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SYCL_INCLUDE_DIR})
set_property(TARGET torch::syclrt APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${PYTORCH_SYCL_LIBRARIES})

set(SYCL_COMPILER_VERSION)
file(READ ${SYCL_VERSION} version_contents)
string(REGEX MATCHALL "__SYCL_COMPILER_VERSION +[0-9]+" VERSION_LINE "${version_contents}")
list(LENGTH VERSION_LINE ver_line_num)
if(${ver_line_num} EQUAL 1)
  string(REGEX MATCHALL "[0-9]+" SYCL_COMPILER_VERSION "${VERSION_LINE}")
endif()

# offline compiler of SYCL compiler
set(IGC_OCLOC_VERSION)
find_program(OCLOC_EXEC ocloc)
if(OCLOC_EXEC)
  set(drv_ver_file "${PROJECT_BINARY_DIR}/OCL_DRIVER_VERSION")
  file(REMOVE ${drv_ver_file})
  execute_process(COMMAND ${OCLOC_EXEC} query OCL_DRIVER_VERSION WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  if(EXISTS ${drv_ver_file})
    file(READ ${drv_ver_file} drv_ver_contents)
    string(STRIP "${drv_ver_contents}" IGC_OCLOC_VERSION)
  endif()
endif()

find_package(SYCL REQUIRED)
if(NOT SYCL_FOUND)
  message("Cannot find SYCL cmake helpers module!")
  return()
endif()

set(PYTORCH_FOUND_XPU TRUE)

message(STATUS "XPU found")
