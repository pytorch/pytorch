#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/test/cuda_lowfp_test.cuh>

#include <assert.h>

using namespace at;

__global__ void kernel(){
  test<Half>();

  __half a = __float2half(3.0f);
  __half b = __float2half(2.0f);
  __half c = a - Half(b);
  assert(static_cast<Half>(c) == Half(1.0));
}

void launch_function(){
  kernel<<<1, 1>>>();
}

// half common math functions tests in device
TEST(HalfCuda, HalfCuda) {
  if (!at::cuda::is_available()) return;
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}
