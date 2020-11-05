#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void igammac_kernel_cuda(TensorIterator& iter) {

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return calc_igammac(a, b);
    });
  });
}

REGISTER_DISPATCH(igammac_stub, &igammac_kernel_cuda);

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

}} // namespace at::native
