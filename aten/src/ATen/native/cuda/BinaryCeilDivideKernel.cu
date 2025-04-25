#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/NumericUtils.h>

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
        // For integers, we need to handle the ceiling division manually
        if (std::is_unsigned<scalar_t>::value || (a >= 0 && b > 0) || (a <= 0 && b < 0)) {
          // For unsigned types or when signs match, use ceiling division formula
          return (a + b - 1) / b;
        } else {
          // For different signs, regular division gives the ceiling result
          return a / b;
        }
      }
    });
  });
}

// Register the CUDA implementation with the dispatcher
REGISTER_DISPATCH(div_ceil_stub, &div_ceil_kernel_cuda);

// Do not add additional kernel definitions to this file to avoid long compile times

} // namespace at::native 