#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void maximum_kernel_cuda(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "maximum_cuda", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return a >= b ? a : b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "maximum_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        // If one of the elements being compared is a NaN, then that element is returned.
        if (a != a) {
          return a;
        }
        if (b != b) {
          return b;
        }
        return a >= b ? a : b;
      });
    });
  }
}

void minimum_kernel_cuda(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "minimum_cuda", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return a <= b ? a : b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "minimum_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        // If one of the elements being compared is a NaN, then that element is returned.
        if (a != a) {
          return a;
        }
        if (b != b) {
          return b;
        }
        return a <= b ? a : b;
      });
    });
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda);
REGISTER_DISPATCH(minimum_stub, &minimum_kernel_cuda);

}} // namespace at::native
