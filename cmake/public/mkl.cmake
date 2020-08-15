find_package(MKL QUIET)

if(NOT TARGET caffe2::mkl)
  add_library(caffe2::mkl INTERFACE IMPORTED)
endif()

set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKL_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKL_LIBRARIES})
