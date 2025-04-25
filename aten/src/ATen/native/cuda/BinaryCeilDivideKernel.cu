#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/NumericUtils.h>
#include <type_traits>

namespace at::native {

// CUDA implementation of ceiling division
void div_ceil_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_ceil_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      // Handle division by zero
      if (b == 0) {
        return 0;
      }
      
      if (std::is_floating_point<scalar_t>::value) {
        // For floating point, use std::ceil(a / b)
        return std::ceil(a / b);
      } else {
        // For integer types
        if (std::is_unsigned<scalar_t>::value) {
          // For unsigned types, always use ceiling division formula
          return (a + b - 1) / b;
        } else {
          // For signed types, check sign relationship
          const bool a_negative = a < 0;
          const bool b_negative = b < 0;
          
          if (a_negative != b_negative) {
            // Different signs - use regular division
            return a / b;
          } else {
            // Same signs - use ceiling division formula
            return (a + b - 1) / b;
          }
        }
      }
    });
  });
}

// Register the CUDA implementation with the dispatcher
REGISTER_DISPATCH(div_ceil_stub, &div_ceil_kernel_cuda);

// Do not add additional kernel definitions to this file to avoid long compile times

} // namespace at::native 