#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <ATen/MetaFunctions.h>

namespace at {
namespace meta {

static std::array<int64_t, 3> upsample_nearest1d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
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

  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  return {nbatch, channels, output_width};
}

TORCH_META_FUNC(upsample_nearest1d) (
  const Tensor& input, IntArrayRef output_size, c10::optional<double> scales
) {
  auto full_output_size = upsample_nearest1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output(full_output_size, input.options());
}

TORCH_META_FUNC(upsample_nearest1d_backward) (
  const Tensor& grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales
) {
  auto full_output_size = upsample_nearest1d_common_check(input_size, output_size);

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output(input_size, grad_output.options());
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

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

// vec variants

Tensor upsample_nearest1d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::upsample_nearest1d(input, osize, scale_w);
}

Tensor upsample_nearest1d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  return at::upsample_nearest1d_backward(grad_output, osize, input_size, scale_w);
}

DEFINE_DISPATCH(upsample_nearest1d_kernel);
DEFINE_DISPATCH(upsample_nearest1d_backward_kernel);

} // namespace native

} // namespace at
