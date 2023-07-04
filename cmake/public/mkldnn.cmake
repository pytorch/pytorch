set(MKLDNN_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})

if(CPU_AARCH64)
  include(${CMAKE_CURRENT_LIST_DIR}/ComputeLibrary.cmake)
endif()

if(USE_SYSTEM_MKLDNN)
  find_package(DNNL REQUIRED)
  set(MKLDNN_FOUND TRUE)
  set(MKLDNN_INCLUDE_DIR)
  set(MKLDNN_LIBRARIES)

  get_property(DNNL_INCLUDE_DIR TARGET DNNL::dnnl PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND MKLDNN_INCLUDE_DIR ${DNNL_INCLUDE_DIR})
  list(APPEND MKLDNN_LIBRARIES DNNL::dnnl)

  find_path(IDEEP_INCLUDE_DIR ideep.hpp
    PATHS "${PROJECT_SOURCE_DIR}/third_party/ideep"
    PATH_SUFFIXES include)
  list(APPEND MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})
else()
  find_package(MKLDNN QUIET)
endif()

if(NOT TARGET caffe2::mkldnn)
  add_library(caffe2::mkldnn INTERFACE IMPORTED)
endif()

set_property(
  TARGET caffe2::mkldnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKLDNN_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkldnn PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKLDNN_LIBRARIES})
if(BUILD_ONEDNN_GRAPH)
  if(NOT TARGET caffe2::dnnl_graph)
    add_library(caffe2::dnnl_graph INTERFACE IMPORTED)
  endif()

  set_property(
    TARGET caffe2::dnnl_graph PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${MKLDNN_INCLUDE_DIR})
  set_property(
    TARGET caffe2::dnnl_graph PROPERTY INTERFACE_LINK_LIBRARIES
    ${MKLDNN_LIBRARIES})
endif()
