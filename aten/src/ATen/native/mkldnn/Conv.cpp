#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt,
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

#define MKLDNNTensor(itensor, options)                                  \
  new_with_itensor_mkldnn(                                              \
      std::move(itensor),                                               \
      optTypeMetaToScalarType(options.dtype_opt()),                     \
      options.device_opt())

// Note [MKLDNN Convolution Memory Formats]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MKLDNN has 3 types of memory formats in convolution:
//
// In case memory format passed from PyTorch (aka. user layout)
// differs from the internal layout which MKLDNN used, a `reorder` is needed;
// otherwise when user layout is identical to internal layout,
// MKLDNN uses a memory `view` upon an existing CPU tensor.
//
// 1. NCHW (CPU tensor, contiguous)
//  input reorder:  NCHW(user) -> Blocked(internal)
//  weight reorder: OIHW(user) -> Blocked(internal)
//  output reorder: Blocked(internal) -> NCHW(user)
//
// 2. NHWC: (CPU tensor, channels last)
//  input view:     NHWC(user) -> NHWC(internal)
//  weight reorder: OHWI(user) -> Blocked(internal)
//  output view:    NHWC(internal) -> NHWC(user)
//
// 3. Blocked (MKLDNN tensor):
//  By explicitly converting a tensor to mkldnn, e.g. `x.to_mkldnn()`,
//  blocked format will propagate between layers. Input, output will be in blocked format.
//
//  For inference case, weight can be prepacked into blocked format by
//  (so as to save weight reoder overhead):
//      model = torch.utils.mkldnn.to_mkldnn(model)
//
//  For training case, grad_output can be CPU tensor or MKLDNN tensor,
//  but weight/bias and grad_weight/grad_bias are always CPU tensor.
//

Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight, const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_convolution: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  bool is_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  auto output_sizes = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
  auto output = at::empty({0}, input.options());

  const ideep::tensor x = itensor_from_tensor(input);
  const ideep::tensor w = itensor_from_tensor(weight);

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, input.suggest_memory_format());
    y = itensor_from_tensor(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::convolution_forward::compute(
        x,
        w,
        b,
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

  if (input.is_mkldnn()) {
    return MKLDNNTensor(y, input.options());
  } else if (!is_channels_last) {
    return mkldnn_to_dense(MKLDNNTensor(y, input.options()));
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  bool is_channels_last = grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  auto grad_input = at::empty({0}, grad_output.options());

  auto grad_y = itensor_from_tensor(grad_output);
  auto w = itensor_view_from_dense(weight);

  ideep::tensor grad_x;
  if (is_channels_last) {
    grad_input.resize_(input_size, grad_output.suggest_memory_format());
    grad_x = itensor_from_tensor(grad_input);
  }
  ideep::convolution_backward_data::compute(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);

  if (grad_output.is_mkldnn()) {
    return MKLDNNTensor(grad_x, grad_output.options());
  } else if (!is_channels_last){
    return mkldnn_to_dense(MKLDNNTensor(grad_x, grad_output.options()));
  } else {
    TORCH_INTERNAL_ASSERT(grad_x.get_desc().is_nhwc());
    return grad_input;
  }
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  bool is_channels_last = grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  const ideep::tensor grad_y = itensor_from_tensor(grad_output);
  const ideep::tensor x = itensor_from_tensor(input);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    ideep::convolution_backward_weights::compute(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }

  if (!is_channels_last) {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())),
        mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())));
  } else {
    return std::make_tuple(
        mkldnn_to_dense(MKLDNNTensor(grad_w, grad_output.options())).to(at::MemoryFormat::ChannelsLast),
        mkldnn_to_dense(MKLDNNTensor(grad_b, grad_output.options())));
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  auto memory_format = input.suggest_memory_format();
  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous(memory_format);

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
