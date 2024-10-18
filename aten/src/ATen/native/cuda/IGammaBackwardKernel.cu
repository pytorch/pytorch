#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

namespace {

void igamma_self_backward_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_self_backward_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t x) -> scalar_t {
      return calc_igamma_grada(a, x);
    });
  });
}

void igammac_self_backward_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_self_backward_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t x) -> scalar_t {
      return calc_igammac_grada(a, x);
    });
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(igamma_self_backward_stub, &igamma_self_backward_kernel_cuda);
REGISTER_DISPATCH(igammac_self_backward_stub, &igammac_self_backward_kernel_cuda);

} // namespace at::native
