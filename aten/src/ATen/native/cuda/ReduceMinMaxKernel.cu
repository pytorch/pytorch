#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <THC/THCNumerics.cuh>
#include <ATen/native/ReduceOps.h>
#include<ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>


namespace at { namespace native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
      return (THCNumerics<acc_t>::isnan(a) || a > b) ? a : b;
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
      return (THCNumerics<acc_t>::isnan(a) || a < b) ? a : b;
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

static void min_kernel_impl(Tensor& result, Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  at::TensorIterator iter = make_reduction("min", result, indice, self, dim, keepdim, self.scalar_type(), kLong);
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(2), "min_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
  });
}

static void max_kernel_impl(Tensor& result, Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  at::TensorIterator iter = make_reduction("max", result, indice, self, dim, keepdim, self.scalar_type(), kLong);
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(2), "max_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MaxOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::lower_bound(), 0));
  });
}

static void _aminmax_kernel_impl(
    Tensor& min_result,
    Tensor& max_result,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  at::TensorIterator iter = make_reduction("_aminmax", min_result,
    max_result, self, dim, keepdim, self.scalar_type());
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, self.scalar_type(), "_aminmax_cuda", [&]() {
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

static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("min_all", result, input, std::vector<int64_t>{}, false, dtype);
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, dtype, "min_all_cuda", [&] {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("max_all", result, input, std::vector<int64_t>{}, false, dtype);
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, dtype, "max_all_cuda", [&] {
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

void _aminmax_all_kernel_impl(Tensor& min_result, Tensor& max_result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("_aminmax_all", min_result, max_result, input,
                             std::vector<int64_t>{}, false, dtype);
  TORCH_CHECK(iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, dtype, "_aminmax_all_cuda", [&] {
    _min_max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);
REGISTER_DISPATCH(min_stub, &min_kernel_impl);
REGISTER_DISPATCH(max_stub, &max_kernel_impl);
REGISTER_DISPATCH(_aminmax_stub, &_aminmax_kernel_impl);
REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl);
REGISTER_DISPATCH(_aminmax_all_stub, &_aminmax_all_kernel_impl);

}} // namespace at::native
