#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void ge_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ge_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "ge_cuda", [&] {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a >= b;
      });
    });
  });
}

REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda);

}} // namespace at::native
