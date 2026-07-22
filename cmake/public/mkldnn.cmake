set(MKLDNN_USE_NATIVE_ARCH ${USE_NATIVE_ARCH})

if(CPU_AARCH64)
  include(${CMAKE_CURRENT_LIST_DIR}/ComputeLibrary.cmake)
endif()

find_package(MKLDNN QUIET)

if(NOT TARGET caffe2::mkldnn)
  add_library(caffe2::mkldnn INTERFACE IMPORTED)
endif()

set_property(
  TARGET caffe2::mkldnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${MKLDNN_INCLUDE_DIR})
set_property(
  TARGET caffe2::mkldnn PROPERTY INTERFACE_LINK_LIBRARIES
  ${MKLDNN_LIBRARIES})

# oneDNN compiles the static dnnl library with -fopenmp but, for the non-SYCL
# OMP runtime, adds no OpenMP link dependency to the target, expecting consumers
# to provide it. PyTorch only propagates Intel's iomp5 via MKL_OPENMP_LIBRARY,
# so a non-MKL OpenMP build leaves nothing linking libomp and binaries that pull
# dnnl objects (e.g. the C++ test executables) fail to resolve omp_*/__kmpc_*
# symbols. Carry the detected OpenMP runtime on the interface to cover that case.
if(MKLDNN_FOUND AND TARGET caffe2::openmp)
  set_property(
    TARGET caffe2::mkldnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES
    caffe2::openmp)
endif()
