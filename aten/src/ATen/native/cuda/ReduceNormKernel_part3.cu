#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/LinearAlgebra.h>
#include <c10/core/Scalar.h>

namespace at::native {

// powsum: computes sum(|x|^p) without the final root
template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type, typename out_t=typename scalar_value_type<scalar_t>::type>
void powsum_kernel_cuda_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<scalar_t, acc_t, out_t, false>(), 0);
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<scalar_t, acc_t, out_t, false>{acc_t(p)}, 0);
  }
}

void powsum_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return powsum_kernel_cuda_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    return powsum_kernel_cuda_impl<at::Half, float, float>(iter, ord);
  }
  else if(iter.dtype(0) == kBFloat16) {
    return powsum_kernel_cuda_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    return powsum_kernel_cuda_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "powsum_cuda", [&] {
    powsum_kernel_cuda_impl<scalar_t>(iter, ord);
  });
}

} // namespace at::native
