#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

void abs_kernel_cuda(TensorIterator& iter) {
  switch (iter.dtype()) {
    case ScalarType::Bool:
      gpu_kernel(iter, []GPU_LAMBDA(bool a) { return a; });
      break;
    case ScalarType::Half:
      gpu_kernel(iter, []GPU_LAMBDA(at::Half a) -> at::Half {
        at::Half nega = -a;
        return a >= nega ? a : nega;
      });
      break;
    default:
      AT_DISPATCH_ALL_TYPES(iter.dtype(), "abs_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return std::abs(a);
        });
      });
  }
}

void bitwise_not_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a) {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);

}}
