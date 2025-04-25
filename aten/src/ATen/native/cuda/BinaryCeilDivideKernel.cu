#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace at::native {

// CUDA implementation of ceiling division
void div_ceil_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_ceil_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      // For integral types
      if (std::is_integral<scalar_t>::value) {
        if (b == 0) {
          return 0; // Handle division by zero for integral types
        }
        // Calculate ceiling division for integers: (a + b - 1) / b
        if ((a < 0) != (b < 0)) {
          // If signs are different, use regular division
          return a / b;
        } else {
          return (a + b - 1) / b;
        }
      } else {
        // For floating point types, use std::ceil(a / b)
        return std::ceil(a / b);
      }
    });
  });
}

// Register the CUDA implementation with the dispatcher
REGISTER_DISPATCH(div_ceil_stub, &div_ceil_kernel_cuda);

// Do not add additional kernel definitions to this file to avoid long compile times

} // namespace at::native 