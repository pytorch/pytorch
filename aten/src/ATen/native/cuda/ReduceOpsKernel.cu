#include <ATen/native/SharedReduceOps.h>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <limits>
#include <tuple>
#include <THC/THCNumerics.cuh>
#include <thrust/tuple.h>
#include <thrust/pair.h>


namespace at { namespace native {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a + b;
  }));
}

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<scalar_t, scalar_t, 2>(iter, WelfordOps<scalar_t, scalar_t, int32_t, float, thrust::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt }, WelfordData<scalar_t, int32_t, float> {});
}

template <>
void std_var_kernel_impl<at::Half>(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<at::Half, at::Half, 2>(iter, WelfordOps<at::Half, float, int32_t, float, thrust::tuple<at::Half, at::Half>> { unbiased, take_sqrt }, WelfordData<float, int32_t, float> {});
}

template <typename scalar_t, typename acc_t=scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(iter, func_wrapper<scalar_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a * b;
  }), 1);
}

static void std_var_kernel_cuda(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
  });
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  float factor = float(iter.num_output_elements()) / iter.numel();
  gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, float> {factor});
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void norm_kernel_cuda_impl(TensorIterator& iter, Scalar val) {
  float p;
  if (val.isIntegral(false)) {
     p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
     p = val.to<acc_t>();
  } else {
     AT_ERROR("norm_kernel_cuda_impl expects norm to be integer or float");
  }

  if (p == static_cast<float>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{ acc_t(p) }, 0);
  }
}

static void sum_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return sum_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::Half, float, float>(iter);
  }
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cuda", [&]() {
    sum_kernel_impl<scalar_t>(iter);
  });
}

static void prod_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return prod_kernel_impl<at::Half, float>(iter);
  }
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "prod_cuda", [&]() {
    prod_kernel_impl<scalar_t>(iter);
  });
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return mean_kernel_impl<at::Half, float, float>(iter);
  }
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mean_cuda", [&]() {
    mean_kernel_impl<scalar_t>(iter);
  });
}

static void norm_kernel_cuda(TensorIterator& iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, p);
  }
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cuda", [&]() {
    norm_kernel_cuda_impl<scalar_t>(iter, p);
  });
}

void and_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a && b;
    }), true);
}

void or_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a || b;
    }), false);
}

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

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda);
REGISTER_DISPATCH(norm_stub, &norm_kernel_cuda);
REGISTER_DISPATCH(and_stub, &and_kernel_cuda);
REGISTER_DISPATCH(or_stub, &or_kernel_cuda);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);

}} // namespace at::native
