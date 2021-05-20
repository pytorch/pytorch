#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void maximum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a || b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::max(a, b);
        }
      });
    });
  }
}

void minimum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "min_elementwise_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::min(a, b);
        }
      });
    });
  }
}

void fmax_kernel_cuda(TensorIterator& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmax_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmax(a, b);
      });
    });
  } else {
    maximum_kernel_cuda(iter);
  }
}

void fmin_kernel_cuda(TensorIterator& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmin_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmin(a, b);
      });
    });
  } else {
    minimum_kernel_cuda(iter);
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda);
REGISTER_DISPATCH(minimum_stub, &minimum_kernel_cuda);
REGISTER_DISPATCH(fmax_stub, &fmax_kernel_cuda);
REGISTER_DISPATCH(fmin_stub, &fmin_kernel_cuda);

}} // namespace at::native
