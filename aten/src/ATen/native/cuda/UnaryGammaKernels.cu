#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/zmath.cuh>

namespace at { namespace native {

void digamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "digamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_digamma(a);
    });
  });
}

void trigamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trigamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_trigamma(a);
    });
  });
}

void polygamma_kernel_cuda(TensorIterator& iter, int64_t n) {
  switch (n) {
    case 0: digamma_kernel_cuda(iter); break;
    case 1: trigamma_kernel_cuda(iter); break;
    default: TORCH_CHECK(false, "polygamma(n,x) is not implemented for n>=2, but was ", n);
  }
}

void lgamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "lgamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::lgamma(a);
    });
  });
}

REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda);
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda);
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda);

}} // namespace at::native
