// Just a little test file to make sure that the CUDA library works

#include <c10/cuda/impl/CUDATest.h>

#include <cuda_runtime.h>

namespace c10 {
namespace cuda {
namespace impl {

int c10_cuda_test() {
  int r;
  cudaGetDevice(&r);
  return r;
}

// This function is not exported
int c10_cuda_private_test() {
  return 2;
}

}}} // namespace c10::cuda::impl
