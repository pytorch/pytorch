// Just a little test file to make sure that the CUDA library works

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/impl/CUDATest.h>

#include <cuda_runtime.h>

namespace c10::cuda::impl {

bool has_cuda_gpu() {
  int count = 0;
  C10_CUDA_IGNORE_ERROR(cudaGetDeviceCount(&count));

  return count != 0;
}

int c10_cuda_test() {
  int r = 0;
  if (has_cuda_gpu()) {
    C10_CUDA_CHECK(cudaGetDevice(&r));
  }
  return r;
}

// This function is not exported
int c10_cuda_private_test() {
  return 2;
}

} // namespace c10::cuda::impl
