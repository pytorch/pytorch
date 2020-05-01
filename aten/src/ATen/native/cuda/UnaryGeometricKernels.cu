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

void acos_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "acos_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::acos(a);
    });
  });
}

void asin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "asin_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::asin(a);
    });
  });
}

void sin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sin_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::sin(a);
    });
  });
}

void cos_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "cos_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::cos(a);
    });
  });
}

void sinh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sinh_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::sinh(a);
    });
  });
}

void cosh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_C10_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "cosh_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::cosh(a);
    });
  });
}

void tanh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Half, iter.dtype(), "tanh_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return std::tanh(a);
    });
  });
}

REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);
REGISTER_DISPATCH(asin_stub, &asin_kernel_cuda);
REGISTER_DISPATCH(sin_stub, &sin_kernel_cuda);
REGISTER_DISPATCH(cos_stub, &cos_kernel_cuda);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel_cuda);
REGISTER_DISPATCH(cosh_stub, &cosh_kernel_cuda);
REGISTER_DISPATCH(tanh_stub, &tanh_kernel_cuda);

}} // namespace at::native
