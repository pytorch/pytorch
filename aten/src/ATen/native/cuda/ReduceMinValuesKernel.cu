#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NumericUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>


namespace at::native {

template <typename acc_t>
struct MinNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
      return (at::_isnan(a) || a < b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t=scalar_t>
void min_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> (MinNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::upper_bound());
}

void min_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cuda", [&]() {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

void min_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
  });
}

void min_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_cuda", [&] {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda)

} // namespace at::native
