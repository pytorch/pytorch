#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace meta {

static std::array<int64_t, 4> upsample_nearest2d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  return {nbatch, channels, output_height, output_width};
}

TORCH_META_FUNC(upsample_nearest2d) (
  const Tensor& input, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  auto full_output_size = upsample_nearest2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || prod_intlist(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output(full_output_size, input.options());
}

TORCH_META_FUNC(upsample_nearest2d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  c10::optional<double> scales_h,
  c10::optional<double> scales_w
) {
  auto full_output_size = upsample_nearest2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output(input_size, grad_output.options());
}

} // namespace meta

namespace native {

TORCH_IMPL_FUNC(upsample_nearest2d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output
) {
  upsample_nearest2d_kernel(kCPU, output, input, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  grad_input.zero_();
  upsample_nearest2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_nearest2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_nearest2d(input, osize, scale_h, scale_w);
}

Tensor upsample_nearest2d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_nearest2d_backward(grad_output, osize, input_size, scale_h, scale_w);
}

DEFINE_DISPATCH(upsample_nearest2d_kernel);
DEFINE_DISPATCH(upsample_nearest2d_backward_kernel);

} // namespace native
} // namespace at
