#include <ATen/Tensor.h>
#include <ATen/Functions.h>

#include <mutex>

namespace at {
namespace cuda {
namespace detail {

__device__ __constant__ float cublas_one_device;
__device__ __constant__ float cublas_zero_device;

float *get_cublas_device_one() {
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    const float one = 1.f;
    cudaMemcpyToSymbol(cublas_one_device, &one, sizeof(float));
  });

  float *ptr;
  cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_one_device);
  return ptr;
}

float *get_cublas_device_zero() {
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    const float zero = 0.f;
    cudaMemcpyToSymbol(cublas_zero_device, &zero, sizeof(float));
  });

  float *ptr;
  cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), cublas_zero_device);
  return ptr;
}

at::Tensor& get_user_alpha_tensor() {
  static at::Tensor alpha;

  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    alpha = at::empty({1}, TensorOptions().device(kCUDA).dtype(kFloat));
  });

  return alpha;
}

} // namespace detail
} // namespace cuda
} // namespace at

