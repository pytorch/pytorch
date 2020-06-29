#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void remainder_kernel_cuda(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "remainder_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        scalar_t r = a % b;
        if ((r != 0) && ((r < 0) != (b < 0))) {
          r += b;
        }
        return r;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "remainder_cuda", [&]() {
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          auto mod = ::fmod(a, b);
          if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;
          return mod;
        });
    });
  }
}

REGISTER_DISPATCH(remainder_stub, &remainder_kernel_cuda);

}} // namespace at::native
