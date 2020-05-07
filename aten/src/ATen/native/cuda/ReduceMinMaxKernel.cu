#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <THC/THCNumerics.cuh>
#include <ATen/native/ReduceOps.h>


namespace at { namespace native {

template <typename scalar_t, typename acc_t=scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return (THCNumerics<acc_t>::isnan(a) || a > b) ? a : b;
    }), at::numeric_limits<acc_t>::lower_bound());
}

template <typename scalar_t, typename acc_t=scalar_t>
void min_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return (THCNumerics<acc_t>::isnan(a) || a < b) ? a : b;
    }), at::numeric_limits<acc_t>::upper_bound());
}

void max_values_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    max_values_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "max_values_cuda", [&]() {
      max_values_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

void min_values_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    min_values_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "min_values_cuda", [&]() {
      min_values_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t>
void argmax_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
    iter,
    ArgMaxOps<acc_t>{},
    thrust::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::lower_bound(), 0));
};

template <typename scalar_t, typename acc_t=scalar_t>
void argmin_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
    iter,
    ArgMinOps<acc_t>{},
    thrust::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmax_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    // Instead of implementing is_nan and warp_shfl_down
    // we can convert halves to float and do all the operations in float
    argmax_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cuda", [&]() {
      argmax_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

void argmin_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    // Instead of implementing is_nan and warp_shfl_down
    // we can convert halves to float and do all the operations in float
    argmin_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_cuda", [&]() {
      argmin_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);

}} // namespace at::native
