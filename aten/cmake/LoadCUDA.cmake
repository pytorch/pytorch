# Detect CUDA architecture and get best NVCC flags
# finding cuda must be first because other things depend on the result
if(NOT NO_CUDA)
  # We set this so that we can use new-style keyword based link libraries
  # with explicit PUBLIC and PRIVATE, so we can control visibility of
  # static libraries we link against.
  set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
  # NB: We MUST NOT run this find_package if NO_CUDA is set, because upstream
  # FindCUDA has a bug where it will still attempt to make use of NOTFOUND
  # compiler variables to run various probe tests.  We could try to fix
  # this, but since FindCUDA upstream is subsumed by first-class support
  # for CUDA language, it seemed not worth fixing.
  find_package(CUDA 8.0 REQUIRED)
endif()
include(CudaCompilerSettings)
include(TestForCudaHalf)
