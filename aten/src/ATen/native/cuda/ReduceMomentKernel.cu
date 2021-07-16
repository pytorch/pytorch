#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

template <typename scalar_t, typename out_t=scalar_t>
void std_var_kernel_impl(TensorIterator& iter, int32_t correction, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using accscalar_t = at::acc_type<scalar_t, true>;
  using ops_t = WelfordOps<scalar_t, accscalar_t, int32_t, float, thrust::pair<out_t, out_t>>;
  gpu_reduce_kernel<scalar_t, out_t, 2>(
      iter, ops_t{correction, take_sqrt}, typename ops_t::acc_t{});
}

static void std_var_kernel_cuda(TensorIterator& iter, int64_t correction, bool take_sqrt) {
  using limits = std::numeric_limits<int32_t>;
  TORCH_CHECK(
      correction < limits::max() && correction > limits::min(),
      "The correction argument for std and var computation on CUDA must "
      "fit within a 32-bit integer, but got ", correction);
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, correction, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, correction, take_sqrt);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    iter.dtype(), "std_cuda", [&]() {
      std_var_kernel_impl<scalar_t>(iter, correction, take_sqrt);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, factor_t> {factor});
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if(iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      mean_kernel_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda);

}} // namespace at::native
