#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>

namespace at { namespace native {

void logical_not_kernel_cuda(TensorIterator& iter) {
  // error check -- this is just ensuring we don't dispatch on types that aren't in ALL_TYPES_AND2(...)
  // so we don't have to maintain a separate list or to do double dispatch.
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_cuda", [&]() {});

  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return !a; });
  });
}

void neg_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "neg_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "neg_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return -a;
      });
    });
  });
}

void sign_kernel_cuda(TensorIterator& iter){
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a){
      return a;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, iter.dtype(), "sign_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            scalar_t zero = scalar_t(0);
            return (zero < a) - (a < zero);
        });
    });
  }
}

void signbit_kernel_cuda(TensorIterator& iter){
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return a < 0; });
  });
}

REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);
REGISTER_DISPATCH(signbit_stub, &signbit_kernel_cuda);

}} // namespace at::native
