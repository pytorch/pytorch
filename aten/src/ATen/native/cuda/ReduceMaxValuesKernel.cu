#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
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

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::lower_bound());
}

void max_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.dtype(), "max_values_cuda", AT_WRAP([&]() {
        max_values_kernel_cuda_impl<scalar_t>(iter);
      }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf, kBool);
}

void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(), "max_cuda", AT_WRAP([&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MaxOps<scalar_t>{},
            thrust::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::lower_bound(), 0));
      }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf, kBool);
}

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_V2(iter.input_dtype(), "max_all_cuda", AT_WRAP([&] {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf, kBool);
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda)

} // namespace at::native
