find_package(MKL QUIET)

if(NOT TARGET caffe2::mkl)
  add_library(caffe2::mkl INTERFACE IMPORTED)
endif()

set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKL_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKL_LIBRARIES} ${MKL_THREAD_LIB})
# TODO: This is a hack, it will not pick up architecture dependent
# MKL libraries correctly; see https://github.com/pytorch/pytorch/issues/73008
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
  ${MKL_ROOT}/lib)
