#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/Reduce.cuh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>

namespace at::native {

template <typename scalar_t, typename acc_t = scalar_t>
void argmin_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMinOps<acc_t>{},
      thrust::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmin_kernel_cuda(TensorIterator& iter) {
  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in
  // float.
  if (iter.dtype(1) == kHalf) {
    argmin_kernel_cuda_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmin_kernel_cuda_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_cuda", [&]() {
      argmin_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda)

} // namespace at::native
