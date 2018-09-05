if(BLAS_SET_BY_USER)
  find_package(MKL REQUIRED)
else()
  find_package(MKL QUIET)
endif()

add_library(caffe2::mkl INTERFACE IMPORTED)
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKL_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkl PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKL_LIBRARIES})
