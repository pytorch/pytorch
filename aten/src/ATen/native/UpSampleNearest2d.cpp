#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

static void upsample_nearest2d_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());

  AT_ASSERT(input_width > 0 && output_width > 0);
  upsample_nearest2d_kernel(kCPU, output, input, scales_h, scales_w);
}

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

Tensor& upsample_nearest2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest2d_out_cpu_template(output, input, output_size, scales_h, scales_w);
  return output;
}

Tensor upsample_nearest2d_cpu(const Tensor& input, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  upsample_nearest2d_out_cpu_template(output, input, output_size, scales_h, scales_w);
  return output;
}

Tensor& upsample_nearest2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  return grad_input;
}

Tensor upsample_nearest2d_backward_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_nearest2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  return grad_input;
}

DEFINE_DISPATCH(upsample_nearest2d_kernel);
DEFINE_DISPATCH(upsample_nearest2d_backward_kernel);

} // namespace native
} // namespace at
