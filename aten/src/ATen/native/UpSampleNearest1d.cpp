#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

static void upsample_nearest1d_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_width = input.size(2);

  upsample_1d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_width,
      output_width);

  output.resize_({nbatch, channels, output_width});

  AT_ASSERT(input_width > 0 && output_width > 0);
  upsample_nearest1d_kernel(kCPU, output, input, scales);
}

static void upsample_nearest1d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  upsample_1d_shape_check(
      Tensor(),
      grad_output,
      nbatch,
      channels,
      input_width,
      output_width);

  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  upsample_nearest1d_backward_kernel(kCPU, grad_input, grad_output, scales);
}
} // namespace

Tensor& upsample_nearest1d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  upsample_nearest1d_out_cpu_template(output, input, output_size, scales);
  return output;
}

Tensor upsample_nearest1d_cpu(const Tensor& input, IntArrayRef output_size, c10::optional<double> scales) {
  auto output = at::empty({0}, input.options());
  upsample_nearest1d_out_cpu_template(output, input, output_size, scales);
  return output;
}

Tensor& upsample_nearest1d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  upsample_nearest1d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, scales);
  return grad_input;
}

Tensor upsample_nearest1d_backward_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_nearest1d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, scales);
  return grad_input;
}

DEFINE_DISPATCH(upsample_nearest1d_kernel);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel);

} // namespace native
} // namespace at
