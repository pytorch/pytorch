#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

ideep::tensor _mkldnn_convolution(
    const ideep::tensor& x,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  auto kernel_size = w.get_dims();

  std::vector<int64_t> input_size = x.get_dims();
  std::vector<int64_t> output_sizes =
      conv_output_size(input_size, kernel_size, padding, stride, dilation);

  ideep::tensor y;
  if (b.has_value()) {
    ideep::convolution_forward::compute(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  }
  return y;
}

Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_convolution: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);
  const ideep::tensor mkldnn_weight = itensor_from_tensor(weight);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  ideep::tensor mkldnn_output = _mkldnn_convolution(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      dilation,
      groups);

  if (input.is_mkldnn()) {
    return new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                   input.options().device_opt());
  } else {
    return mkldnn_to_dense(
        new_with_itensor_mkldnn(std::move(mkldnn_output), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()));
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  auto mkldnn_grad_output = itensor_from_tensor(grad_output);
  auto mkldnn_weight = itensor_from_tensor(weight);

  ideep::tensor mkldnn_grad_input;
  ideep::convolution_backward_data::compute(
      mkldnn_grad_output,
      mkldnn_weight,
      input_size.vec(),
      mkldnn_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);

  return mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_input),
                                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                                 grad_output.options().device_opt()));
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  const ideep::tensor mkldnn_grad_output = itensor_from_tensor(grad_output);
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);

  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        weight_size.vec(),
        mkldnn_grad_weight,
        mkldnn_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        weight_size.vec(),
        mkldnn_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }

  return std::make_tuple(
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_weight),
                                              optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                              grad_output.options().device_opt())),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(mkldnn_grad_bias),
                                              optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                              grad_output.options().device_opt())));
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}}  // namespace at::native

#endif
