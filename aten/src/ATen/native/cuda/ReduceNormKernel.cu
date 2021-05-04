#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_cuda_impl(TensorIterator& iter, const Scalar& val) {
  double p;
  if (val.isIntegral(false)) {
     p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
     p = val.to<double>();
  } else {
     AT_ERROR("norm_kernel_cuda_impl expects norm to be integer or float");
  }

  if (p == static_cast<double>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<scalar_t, acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<scalar_t, acc_t>{ acc_t(p) }, 0);
  }
}

static void norm_kernel_cuda(TensorIterator& iter, const Scalar& p) {
  if (iter.input_dtype() == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.input_dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, p);
  }
  else if(iter.input_dtype() == kBFloat16) {
    return norm_kernel_cuda_impl<at::BFloat16, float>(iter, p);
  } else if (iter.dtype(1) == kBFloat16 && iter.input_dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::BFloat16, float, float>(iter, p);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cuda", [&] {
    norm_kernel_cuda_impl<scalar_t>(iter, p);
  });
}

REGISTER_DISPATCH(norm_stub, &norm_kernel_cuda);

}} // namespace at::native
