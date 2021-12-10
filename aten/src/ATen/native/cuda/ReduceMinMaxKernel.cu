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


namespace at { namespace native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
      return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t=scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> (MaxNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::lower_bound());
}

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

void max_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cuda", [&]() {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

void min_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cuda", [&]() {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
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
  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in float.
  if (iter.dtype(1) == kHalf) {
    argmax_kernel_cuda_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmax_kernel_cuda_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cuda", [&]() {
      argmax_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

void argmin_kernel_cuda(TensorIterator& iter) {
  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in float.
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

void min_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
  });
}

void max_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MaxOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::lower_bound(), 0));
  });
}

void aminmax_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      thrust::pair<scalar_t, scalar_t>(
        at::numeric_limits<scalar_t>::upper_bound(),
        at::numeric_limits<scalar_t>::lower_bound()
      )
    );
  });
}

void min_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_cuda", [&] {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_all_cuda", [&] {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

template <typename scalar_t>
void _min_max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, MinMaxOps<scalar_t, scalar_t, int32_t>{}, thrust::pair<scalar_t, scalar_t>(
      at::numeric_limits<scalar_t>::upper_bound(),
      at::numeric_limits<scalar_t>::lower_bound()
  ));
}

void aminmax_allreduce_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_all_cuda", [&] {
    _min_max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);

}} // namespace at::native
