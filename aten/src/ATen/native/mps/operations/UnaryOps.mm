//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/ScanKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/MPSFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/acos_native.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/cosh_native.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/exp2_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {

enum class MPSCumulativeOpType : uint8_t {
  CUMSUM = 0,
  CUMPROD = 1,
};

static void cumulative_op_impl(const Tensor& self,
                               int64_t dim,
                               std::optional<ScalarType> dtype,
                               const Tensor& result,
                               MPSCumulativeOpType cumulativeOpType) {
  auto nDims = self.dim();
  auto wrapped_dim = maybe_wrap_dim(dim, nDims);
  TORCH_CHECK(wrapped_dim >= 0 && wrapped_dim < std::max(1LL, self.ndimension()),
              "Expected wrapped dim to be between 0 and ",
              self.ndimension(),
              " but got ",
              wrapped_dim,
              "(original dim is ",
              dim,
              ")");
  auto input = dtype.has_value() ? self.to(dtype.value()) : self;
  const std::string scan_op_name = cumulativeOpType == MPSCumulativeOpType::CUMSUM ? "cumsum" : "cumprod";

  if (input.is_complex()) {
    if (cumulativeOpType == MPSCumulativeOpType::CUMSUM) {
      auto input_real = at::view_as_real(input.dim() == 0 ? input.view({1}) : input);
      auto result_real = at::view_as_real(result.dim() == 0 ? result.view({1}) : result);
      return cumulative_op_impl(input_real, wrapped_dim, std::nullopt, result_real, MPSCumulativeOpType::CUMSUM);
    }
    auto input_view = input.dim() == 0 ? input.view({1}) : input;
    auto result_view = result.dim() == 0 ? result.view({1}) : result;
    return mps::scan_simple_mps_impl(input_view, result_view, wrapped_dim, scan_op_name);
  }

  // cumsum/cumprod of a 0-dim tensor is the identity.
  if (input.dim() == 0) {
    result.copy_(input);
    return;
  }

  const bool castInputData = isIntegralType(input.scalar_type(), /*includeBool=*/true) &&
      input.scalar_type() != ScalarType::Int && input.scalar_type() != ScalarType::Long;
  if (castInputData) {
    auto input_i32 = input.to(ScalarType::Int);
    if (result.scalar_type() == ScalarType::Long) {
      mps::scan_simple_mps_impl(input_i32, result, wrapped_dim, scan_op_name);
    } else {
      auto result_i32 = at::empty(result.sizes(), result.options().dtype(ScalarType::Int));
      mps::scan_simple_mps_impl(input_i32, result_i32, wrapped_dim, scan_op_name);
      result.copy_(result_i32);
    }
    return;
  }
  // int32 -> int64 result: scan int32 directly; the kernel widens (no upcast).
  if (input.scalar_type() == ScalarType::Int && result.scalar_type() == ScalarType::Long) {
    mps::scan_simple_mps_impl(input, result, wrapped_dim, scan_op_name);
    return;
  }

  // int64 input, or a dtype= override: match the scan input dtype to the result.
  auto scan_input = input.scalar_type() == result.scalar_type() ? input : input.to(result.scalar_type());

  mps::scan_simple_mps_impl(scan_input, result, wrapped_dim, scan_op_name);
}

TORCH_IMPL_FUNC(cumsum_out_mps)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMSUM);
}

TORCH_IMPL_FUNC(cumprod_out_mps)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, const Tensor& result) {
  return cumulative_op_impl(self, dim, dtype, result, MPSCumulativeOpType::CUMPROD);
}

} // namespace at::native
