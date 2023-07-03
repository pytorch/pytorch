#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/UpSample.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#endif

namespace at {
namespace meta {

TORCH_META_FUNC(upsample_nearest1d) (
    const Tensor& input, IntArrayRef output_size, c10::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options());
}

TORCH_META_FUNC(_upsample_nearest_exact1d) (
  const Tensor& input, IntArrayRef output_size, c10::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options());
}

TORCH_META_FUNC(upsample_nearest1d_backward) (
    const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

TORCH_META_FUNC(_upsample_nearest_exact1d_backward) (
  const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales
) {
  auto full_output_size = native::upsample_1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace meta


namespace native {

TORCH_IMPL_FUNC(upsample_nearest1d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_kernel(kCPU, output, input, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    const Tensor& output
) {
  _upsample_nearest_exact1d_kernel(kCPU, output, input, scales);
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    const Tensor& grad_input
) {
  grad_input.zero_();
  upsample_nearest1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    const Tensor& grad_input
) {
  grad_input.zero_();
  _upsample_nearest_exact1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// vec variants

Tensor upsample_nearest1d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::upsample_nearest1d(input, osize, scale_w);
}

Tensor _upsample_nearest_exact1d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::_upsample_nearest_exact1d(input, osize, scale_w);
}

DEFINE_DISPATCH(upsample_nearest1d_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_kernel);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel);
DEFINE_DISPATCH(_upsample_nearest_exact1d_backward_kernel);

} // namespace native

} // namespace at
