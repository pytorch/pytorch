set(MKLDNN_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})

if(CPU_AARCH64)
  include(${CMAKE_CURRENT_LIST_DIR}/ComputeLibrary.cmake)
endif()

find_package(MKLDNN QUIET)

if(NOT TARGET torch::mkldnn)
  add_library(torch::mkldnn INTERFACE IMPORTED)
endif()

set_property(
  TARGET torch::mkldnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKLDNN_INCLUDE_DIR})
set_property(
  TARGET torch::mkldnn PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKLDNN_LIBRARIES})
