#include <ATen/native/NaiveDilatedConvolution.h>

namespace at {
namespace native {

Tensor slow_conv_dilated2d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor undefined;
  internal::slow_conv_dilated_shape_check<2>(
      input,
      weight,
      bias,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 4;
  auto options = input.options();
  // calculate output tensor size
  auto output_size = internal::get_output_size<2>(
      input, weight, kernel_size, stride_size, pad_size, dilation_size);
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
  Tensor output = at::empty(output_size, options);
  Tensor output_ = (is_batch ? output : output.unsqueeze(0));

  slow_conv_dilated_all_cpu_template<2>(
      output_,
      input_,
      weight_,
      bias_,
      undefined,
      undefined,
      undefined,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

Tensor slow_conv_dilated3d_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  Tensor undefined;
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      bias,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 5;
  auto options = input.options();
  // calculate output tensor size
  auto output_size = internal::get_output_size<3>(
      input, weight, kernel_size, stride_size, pad_size, dilation_size);
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
  Tensor output = at::empty(output_size, options);
  Tensor output_ = (is_batch ? output : output.unsqueeze(0));

  slow_conv_dilated_all_cpu_template<3>(
      output,
      input_,
      weight_,
      bias_,
      undefined,
      undefined,
      undefined,
      undefined,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return output;
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size,
    const std::array<bool, 3ul> output_mask) {
  Tensor undefined;
  internal::slow_conv_dilated_shape_check<3>(
      input,
      weight,
      undefined,
      grad_output,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  auto is_batch = input.dim() == 5;
  auto options = grad_output.options();
  // template function assumes batched tensors.  unsqueeze(0) will
  // insert batch dimension without affecting the original tensor.
  const Tensor grad_output_ =
      (is_batch ? grad_output.contiguous()
                : grad_output.contiguous().unsqueeze(0));
  const Tensor input_ =
      (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  const Tensor weight_ = weight.contiguous();
  // compute only gradients for which the corresponding output_mask is true:
  Tensor grad_input =
      (output_mask[0] ? at::empty(input.sizes(), options) : undefined);
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
  Tensor grad_bias =
      (output_mask[2] ? at::empty(weight.size(0), options) : undefined);
  Tensor grad_input_ =
      (output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
                      : undefined);
  slow_conv_dilated_all_cpu_template<3>(
      undefined,
      input_,
      weight_,
      undefined,
      grad_output_,
      grad_input,
      grad_weight,
      grad_bias,
      kernel_size,
      stride_size,
      pad_size,
      dilation_size);
  return std::tie(grad_input, grad_weight, grad_bias);
}

} // namespace native
} // namespace at
