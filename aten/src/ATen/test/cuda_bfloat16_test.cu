#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/test/cuda_lowfp_test.cuh>

#include <assert.h>

using namespace at;

__global__ void kernel(){
  test<BFloat16>();

  __nv_bfloat16 a = __float2bfloat16(3.0f);
  __nv_bfloat16 b = __float2bfloat16(2.0f);
  __nv_bfloat16 c = a - BFloat16(b);
  assert(static_cast<BFloat16>(c) == BFloat16(1.0));
}

void launch_function(){
  kernel<<<1, 1>>>();
}

// bfloat16 common math functions tests in device
TEST(BFloat16Cuda, BFloat16Cuda) {
  if (!at::cuda::is_available()) return;
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}
#endif
