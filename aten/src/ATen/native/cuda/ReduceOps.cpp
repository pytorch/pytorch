#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ReduceOps.h>

#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/TensorIterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {
namespace {

void norm_kernel_cuda(TensorIterator& iter, const Scalar& val) {
  double p = 0;
  if (val.isIntegral(false)) {
    p = static_cast<double>(val.to<int64_t>());
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(false, "norm_kernel_cuda_impl expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }

  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }

}

void min_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  min_launch_kernel(iter);
}

void max_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  max_launch_kernel(iter);
}

void aminmax_kernel_impl(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& min_result, Tensor& max_result) {
  at::TensorIterator iter = make_reduction("aminmax_cuda", min_result,
                                           max_result, self, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    aminmax_launch_kernel(iter);
  }
}

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("min_all", result, input, IntArrayRef{}, false, dtype);
  min_all_launch_kernel(iter);
}

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("max_all", result, input, IntArrayRef{}, false, dtype);
  max_all_launch_kernel(iter);
}

void aminmax_allreduce_kernel_impl(const Tensor& input, Tensor& min_result, Tensor& max_result) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("aminmax_cuda", min_result, max_result, input,
                             IntArrayRef{}, false, dtype);
  TORCH_CHECK(iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  aminmax_allreduce_launch_kernel(iter);
}

}  // namespace (anonymous)

REGISTER_CUDA_DISPATCH(min_stub, &min_kernel_impl)
REGISTER_CUDA_DISPATCH(max_stub, &max_kernel_impl)
REGISTER_CUDA_DISPATCH(min_all_stub, &min_all_kernel_impl)
REGISTER_CUDA_DISPATCH(max_all_stub, &max_all_kernel_impl)
REGISTER_CUDA_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel_impl)
REGISTER_CUDA_DISPATCH(aminmax_stub, &aminmax_kernel_impl)

REGISTER_CUDA_DISPATCH(norm_stub, &norm_kernel_cuda)

} // namespace at::native
