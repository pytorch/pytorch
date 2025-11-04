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

template <typename scalar_t>
void _min_max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      thrust::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

void aminmax_allreduce_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(), "aminmax_all_cuda", AT_WRAP([&] {
        _min_max_values_kernel_cuda_impl<scalar_t>(iter);
      }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf, kBool);
}

void aminmax_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_V2(
      iter.input_dtype(), "aminmax_cuda", AT_WRAP([&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MinMaxOps<scalar_t, scalar_t, int32_t>{},
            thrust::pair<scalar_t, scalar_t>(
                at::numeric_limits<scalar_t>::upper_bound(),
                at::numeric_limits<scalar_t>::lower_bound()));
      }), AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kBFloat16, kHalf, kBool);
}

} // namespace at::native
