#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

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
  } else if (p == static_cast<float>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{ acc_t(p) }, 0);
  }
}

static void norm_kernel_cuda(TensorIterator& iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, p);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if(iter.dtype() == kBFloat16) {
    return norm_kernel_cuda_impl<at::BFloat16, float>(iter, p);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::BFloat16, float, float>(iter, p);
  }
  #endif
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cuda", [&]() {
    norm_kernel_cuda_impl<scalar_t>(iter, p);
  });
}

REGISTER_DISPATCH(norm_stub, &norm_kernel_cuda);

}} // namespace at::native
