#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

namespace {

void igamma_grada_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_grada_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t x) -> scalar_t {
      return calc_igamma_grada<scalar_t, /*is_cuda=*/true>(a, x);
    });
  });
}

void igammac_grada_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_grada_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t x) -> scalar_t {
      return calc_igammac_grada<scalar_t, /*is_cuda=*/true>(a, x);
    });
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(igamma_grada_stub, &igamma_grada_kernel_cuda);
REGISTER_DISPATCH(igammac_grada_stub, &igammac_grada_kernel_cuda);

} // namespace at::native
