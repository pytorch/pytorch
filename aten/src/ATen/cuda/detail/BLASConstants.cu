#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/Exceptions.h>

#include <mutex>

namespace at {
namespace cuda {
namespace detail {

__device__ __constant__ float cublas_one_device;
__device__ __constant__ float cublas_zero_device;

float *get_cublas_device_one() {
  static c10::once_flag init_flag;

  c10::call_once(init_flag, []() {
    const float one = 1.f;
    AT_CUDA_CHECK(cudaMemcpyToSymbol(cublas_one_device, &one, sizeof(float)));
  });

  float *ptr;
  AT_CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_one_device));
  return ptr;
}

float *get_cublas_device_zero() {
  static c10::once_flag init_flag;

  c10::call_once(init_flag, []() {
    const float zero = 0.f;
    AT_CUDA_CHECK(cudaMemcpyToSymbol(cublas_zero_device, &zero, sizeof(float)));
  });

  float *ptr;
  AT_CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_zero_device));
  return ptr;
}

float *get_user_alpha_ptr() {
  static float *alpha_ptr;

  static c10::once_flag init_flag;

  c10::call_once(init_flag, []() {
    AT_CUDA_CHECK(cudaMalloc(&alpha_ptr, sizeof(float)));
  });

  return alpha_ptr;
}

} // namespace detail
} // namespace cuda
} // namespace at
