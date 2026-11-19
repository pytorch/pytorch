#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_inter_area2d.h>
#include <ATen/ops/_upsample_inter_area2d_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(_upsample_inter_area2d) (
  const Tensor& input, IntArrayRef output_size, std::optional<double> scales_h, std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

} // namespace at::meta
namespace at::native {

TORCH_IMPL_FUNC(_upsample_inter_area2d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  _upsample_inter_area2d_kernel(kCPU, output, input, scales_h, scales_w);
}

// vec variant

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor _upsample_inter_area2d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::_upsample_inter_area2d(input, osize, scale_h, scale_w);
}

DEFINE_DISPATCH(_upsample_inter_area2d_kernel);

} // namespace at::native
