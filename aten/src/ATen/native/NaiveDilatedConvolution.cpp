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

} // namespace native
} // namespace at
