#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/test/cuda_lowfp_test.cuh>

#include <assert.h>

using namespace at;

__global__ void kernel(){
  test<Half>();

  // test complex<32>
  Half real = 3.0f;
  Half imag = -10.0f;
  auto complex = c10::complex<Half>(real, imag);
  assert(complex.real() == real);
  assert(complex.imag() == imag);
}

__global__ void kernel(){
  test();
}

void launch_function(){
  kernel<<<1, 1>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// half common math functions tests in device
TEST(HalfCuda, HalfCuda) {
  if (!at::cuda::is_available()) return;
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}
