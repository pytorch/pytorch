#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <limits>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void logical_xor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_xor_cuda", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(2), "logical_xor_cuda", [&]() {
      using other_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_xor_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(self_t a, other_t b) -> scalar_t {
          return static_cast<scalar_t>(bool(a) != bool(b));
        });
      });
    });
  });
}

REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_cuda);

}} // namespace at::native
