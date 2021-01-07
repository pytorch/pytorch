#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace meta {

static std::array<int64_t, 4> upsample_nearest2d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];
  int64_t output_height = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];
  int64_t input_height = input_size[3];

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

  return {nbatch, channels, output_width, output_height};
}

TORCH_META_FUNC(upsample_nearest2d) (
  const Tensor& input, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  auto full_output_size = upsample_nearest2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.numel() != 0 ||
        (input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0)
        ) &&
      input.dim() == 4,
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output(full_output_size, input.options());
}

TORCH_META_FUNC(upsample_nearest2d_backward) (
  const Tensor& grad_output, IntArrayRef input_size, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  auto full_output_size = upsample_nearest2d_common_check(input_size, output_size);

  check_dim_size(grad_output, 4, 0, full_output_size[0]);
  check_dim_size(grad_output, 4, 1, full_output_size[1]);
  check_dim_size(grad_output, 4, 2, full_output_size[2]);
  check_dim_size(grad_output, 4, 3, full_output_size[3]);

  set_output(input_size, grad_output.options());
}

} // namespace meta

namespace native {
namespace {

static void upsample_nearest2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
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

  upsample_2d_shape_check(
      Tensor(),
      grad_output,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  upsample_nearest2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);
}
} // namespace

TORCH_IMPL_FUNC(upsample_nearest2d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output
) {
  upsample_nearest2d_kernel(kCPU, output, input, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  upsample_nearest2d_backward_kernel(kCPU, grad_input, grad_output, scales_h, scales_w);
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_nearest2d_cpu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_nearest2d(input, osize, scale_h, scale_w);
}

Tensor upsample_nearest2d_backward_cpu(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_nearest2d_backward_out_cpu_template(
      grad_input, grad_output, osize, input_size, scale_h, scale_w);
  return grad_input;
}

DEFINE_DISPATCH(upsample_nearest2d_kernel);
DEFINE_DISPATCH(upsample_nearest2d_backward_kernel);

} // namespace native
} // namespace at
