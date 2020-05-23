#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/zmath.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void max_elementwise_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a || b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        // isnan(half) breaks the Windows build. We explicitly cast half to float.
        using acc_type = typename AccumulateType<scalar_t, /*is_cuda=*/true>::type;
        // We avoid using nan or nanf because we want to return the same type as scalar_t.
        if (::isnan(static_cast<acc_type>(a))) {
          return a;
        } else if (::isnan(static_cast<acc_type>(b))) {
          return b;
        } else {
          return ::max(a, b);
        }
      });
    });
  }
}

void min_elementwise_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "min_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "min_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        // isnan(half) breaks the Windows build. We explicitly cast half to float.
        using acc_type = typename AccumulateType<scalar_t, /*is_cuda=*/true>::type;
        // We avoid using nan or nanf because we want to return the same type as scalar_t.
        if (::isnan(static_cast<acc_type>(a))) {
          return a;
        } else if (::isnan(static_cast<acc_type>(b))) {
          return b;
        } else {
          return ::min(a, b);
        }
      });
    });
  }
}

REGISTER_DISPATCH(max_elementwise_stub, &max_elementwise_kernel_cuda);
REGISTER_DISPATCH(min_elementwise_stub, &min_elementwise_kernel_cuda);

}} // namespace at::native
