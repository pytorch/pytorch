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
  } else if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "maximum_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        // isnan(half) breaks the Windows build. We explicitly cast half to float.
        using acc_type = typename AccumulateType<scalar_t, /*is_cuda=*/true>::type;
        if (::isnan(static_cast<acc_type>(a)) || a >= b) {
          return a;
        }
        return b;
      });
    });
  } else {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "maximum_cuda", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        if (::isnan(a.real())) {
          return a;
        }
        if (::isnan(b.real())) {
          return b;
        }
        if (a.real() > b.real()) {
          return a;
        }
        if (b.real() >= a.real() && b.imag() >= a.imag()) {
          return b;
        }
        return a;
      });
    });
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda);

}} // namespace at::native
