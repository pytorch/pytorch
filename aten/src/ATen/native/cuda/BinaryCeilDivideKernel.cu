#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/NumericUtils.h>

namespace at::native {

// Helper template functions to handle different numeric types
template <typename T, typename = std::enable_if_t<std::is_unsigned<T>::value>>
__host__ __device__ inline T ceil_div_unsigned(T a, T b) {
  // Handle special case to avoid division by zero
  if (b == 0) return 0;
  return (a + b - 1) / b;
}

template <typename T, typename = std::enable_if_t<std::is_signed<T>::value>>
__host__ __device__ inline T ceil_div_signed(T a, T b) {
  // Handle special case to avoid division by zero
  if (b == 0) return 0;
  
  // Different sign case - regular division
  if ((a < 0) != (b < 0)) return a / b;
  
  // Same sign case - ceiling division
  return (a + b - 1) / b;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
__host__ __device__ inline T ceil_div_float(T a, T b) {
  return std::ceil(a / b);
}

// CUDA implementation of ceiling division
void div_ceil_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_ceil_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      if constexpr (std::is_floating_point<scalar_t>::value) {
        return ceil_div_float(a, b);
      } else if constexpr (std::is_unsigned<scalar_t>::value) {
        return ceil_div_unsigned(a, b);
      } else {
        return ceil_div_signed(a, b);
      }
    });
  });
}

// Register the CUDA implementation with the dispatcher
REGISTER_DISPATCH(div_ceil_stub, &div_ceil_kernel_cuda);

// Do not add additional kernel definitions to this file to avoid long compile times

} // namespace at::native 