#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a + b;
  }));
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a * b;
  }), 1);
}

static void sum_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return sum_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::Half, float, float>(iter);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if (iter.dtype() == kBFloat16) {
    return sum_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::BFloat16, float, float>(iter);
  }
  #endif
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, iter.dtype(), "sum_cuda", [&]() {
    sum_kernel_impl<scalar_t>(iter);
  });
}

static void prod_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return prod_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return prod_kernel_impl<at::Half, float, float>(iter);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if (iter.dtype() == kBFloat16) {
    return prod_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return prod_kernel_impl<at::BFloat16, float, float>(iter);
  }
  #endif
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "prod_cuda", [&]() {
    prod_kernel_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);

}} // namespace at::native
