find_package(MKL QUIET)

if(NOT TARGET caffe2::mkl)
  add_library(caffe2::mkl INTERFACE IMPORTED)
endif()

target_link_libraries(caffe2::mkl INTERFACE MKL::MKL)
