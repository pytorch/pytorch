#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>

namespace at::native {

void and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc && static_cast<bool>(val));
            }),
            true);
      });
}

void or_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc || static_cast<bool>(val));
            }),
            false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda)
REGISTER_DISPATCH(or_stub, &or_kernel_cuda)

} // namespace at::native
