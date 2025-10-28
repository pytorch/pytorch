#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/Exceptions.h>

namespace at {
namespace cuda {
namespace detail {

__device__ __constant__ float cublas_one_device{1.f};
__device__ __constant__ float cublas_zero_device{0.f};

float *get_cublas_device_one() {
  float *ptr;
  AT_CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_one_device));
  return ptr;
}

float *get_cublas_device_zero() {
  float *ptr;
  AT_CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_zero_device));
  return ptr;
}

float *get_user_alpha_ptr() {
  static float *alpha_ptr;

  static bool init_flag [[maybe_unused]] = []() {
    AT_CUDA_CHECK(cudaMalloc(&alpha_ptr, sizeof(float)));
    return true;
  }();

  return alpha_ptr;
}

} // namespace detail
} // namespace cuda
} // namespace at
