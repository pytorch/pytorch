#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding_l, IntArrayRef padding_r, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

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

std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding_l, IntArrayRef padding_r, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace {
// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
inline ideep::tensor get_mkldnn_tensor(const at::Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    return at::native::itensor_from_mkldnn(tensor);
  } else {
    return at::native::itensor_view_from_dense(tensor);
  }
}
}

namespace at { namespace native {

ideep::tensor _mkldnn_convolution(
    const ideep::tensor& x,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  auto kernel_size = w.get_dims();

  std::vector<int64_t> input_size = x.get_dims();
  std::vector<int64_t> output_sizes =
      conv_output_size_lr(input_size, kernel_size, padding_l, padding_r, stride, dilation);

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
        {padding_l.begin(), padding_l.end()},
        {padding_r.begin(), padding_r.end()},
        groups);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding_l.begin(), padding_l.end()},
        {padding_r.begin(), padding_r.end()},
        groups);
  }
  return y;
}

Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  const ideep::tensor mkldnn_input = get_mkldnn_tensor(input);
  const ideep::tensor mkldnn_weight = get_mkldnn_tensor(weight);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = get_mkldnn_tensor(bias);
  }

  ideep::tensor mkldnn_output = _mkldnn_convolution(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      padding_l,
      padding_r,
      stride,
      dilation,
      groups);

  auto ret = new_with_itensor_mkldnn(std::move(mkldnn_output), input.options());
  return input.is_mkldnn() ? ret : mkldnn_to_dense(ret);
}

Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  return at::native::mkldnn_convolution(
      input, weight, bias, padding, padding, stride, dilation, groups);
}

static Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding_l, IntArrayRef padding_r, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  auto mkldnn_grad_output = get_mkldnn_tensor(grad_output);
  auto mkldnn_weight = get_mkldnn_tensor(weight);

  ideep::tensor mkldnn_grad_input;
  ideep::convolution_backward_data::compute(
      mkldnn_grad_output,
      mkldnn_weight,
      input_size.vec(),
      mkldnn_grad_input,
      stride.vec(),
      dilation.vec(),
      padding_l.vec(),
      padding_r.vec(),
      groups);

  auto ret = new_with_itensor_mkldnn(std::move(mkldnn_grad_input), grad_output.options());
  return grad_output.is_mkldnn() ? ret : mkldnn_to_dense(ret);
}

static std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding_l, IntArrayRef padding_r, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  const ideep::tensor mkldnn_grad_output = get_mkldnn_tensor(grad_output);
  const ideep::tensor mkldnn_input = get_mkldnn_tensor(input);

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
        padding_l.vec(),
        padding_r.vec(),
        groups);
  } else {
    ideep::convolution_backward_weights::compute(
        mkldnn_input,
        mkldnn_grad_output,
        weight_size.vec(),
        mkldnn_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding_l.vec(),
        padding_r.vec(),
        groups);
  }

  auto grad_weight = new_with_itensor_mkldnn(std::move(mkldnn_grad_weight), grad_output.options());
  auto grad_bias = new_with_itensor_mkldnn(std::move(mkldnn_grad_bias), grad_output.options());
  return std::make_tuple(grad_weight, grad_bias);
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {
  return at::native::mkldnn_convolution_backward_input(
      input_size, grad_output, weight, padding, padding,
      stride, dilation, groups, bias_defined);
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, bool bias_defined) {
  return at::native::mkldnn_convolution_backward_weights(
      weight_size, grad_output, input, padding, padding,
      stride, dilation, groups, bias_defined);
}

std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding_l, IntArrayRef padding_r, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask)
{
  const Tensor& grad_output = grad_output_t.is_mkldnn() ?
      grad_output_t : grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::native::mkldnn_convolution_backward_input(
        input.sizes(), grad_output, weight, padding_l, padding_r,
        stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::native::mkldnn_convolution_backward_weights(
        weight.sizes(), grad_output, input, padding_l, padding_r,
        stride, dilation, groups, output_mask[2]);
    if (!weight.is_mkldnn()) {
      grad_weight = mkldnn_to_dense(grad_weight);
    }
    if (output_mask[2]) {
      grad_bias = mkldnn_to_dense(grad_bias);
    }
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask) {
  return at::native::mkldnn_convolution_backward(
      input, grad_output_t, weight, padding, padding,
      stride, dilation, groups, output_mask);
}

}}  // namespace at::native

#endif
