// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bilinear2d_aa.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_native.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(upsample_bilinear2d) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

TORCH_META_FUNC(upsample_bilinear2d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

TORCH_META_FUNC(_upsample_bilinear2d_aa) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

TORCH_META_FUNC(_upsample_bilinear2d_aa_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options().memory_format(grad_output.suggest_memory_format()));
}

} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(upsample_bilinear2d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  upsample_bilinear2d_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
  grad_input.zero_();
  upsample_bilinear2d_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
}


TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output
) {
  _upsample_bilinear2d_aa_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input
) {
  grad_input.zero_();
  _upsample_bilinear2d_aa_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_bilinear2d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_bilinear2d(input, osize, align_corners, scale_h, scale_w);
}

Tensor _upsample_bilinear2d_aa(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::_upsample_bilinear2d_aa(input, osize, align_corners, scale_h, scale_w);
}

DEFINE_DISPATCH(upsample_bilinear2d_kernel);
DEFINE_DISPATCH(upsample_bilinear2d_backward_kernel);
DEFINE_DISPATCH(_upsample_bilinear2d_aa_kernel);
DEFINE_DISPATCH(_upsample_bilinear2d_aa_backward_kernel);

} // namespace at::native
