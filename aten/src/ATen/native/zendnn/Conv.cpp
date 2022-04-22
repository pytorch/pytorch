#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_ZENDNN_ENABLED()

namespace at {
namespace native {

Tensor zendnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_forward: ATen not compiled with ZENDNN support");
}

Tensor zendnn_convolution_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward_input: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor> zendnn_convolution_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward_weights: ATen not compiled with ZENDNN support");
}

std::tuple<Tensor, Tensor, Tensor> zendnn_convolution_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward: ATen not compiled with ZENDNN support");
}

REGISTER_NO_CPU_DISPATCH(zendnn_convolution_backward_stub);

} // namespace native
} // namespace at

#else // AT_ZENDNN_EBABLED

#include <ATen/native/ConvUtils.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace zendnn {

tensor _zendnn_convolution(
    const tensor& x,
    const tensor& w,
    const c10::optional<tensor>& b,
    c10::IntArrayRef padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups) {
  std::vector<int64_t> output_sizes = at::native::conv_output_size(
      x.get_dims(), w.get_dims(), padding, stride, dilation);
  tensor y;
  const engine& aengine = utils::engine::cpu_engine();
  tensor::desc src_desc, weights_desc, bias_desc;
  primitive_attr op_attr = primitive_attr();
  // make weights and dilates compatible with ZENDNN
  auto weights = w.make_grouped_weights(groups);
  auto dilates =
      utils::get_compatible_dilates({dilation.begin(), dilation.end()});

  // align weights data type with src
  memory::data_type dst_data_type = x.get_data_type() == memory::data_type::bf16
      ? memory::data_type::bf16
      : memory::data_type::f32;
  src_desc = x.get_desc().to_type(dst_data_type);
  weights_desc = weights.get_desc().to_type(dst_data_type);

  if (b.has_value()) {
    bias_desc = b.value().get_desc();
  }

  op_attr.set_scratchpad_mode(scratchpad_mode::user);

  auto dst_desc =
      tensor::desc({output_sizes.cbegin(), output_sizes.cend()}, dst_data_type);

  auto src_desc_query = src_desc.to_format_any();
  auto weights_desc_query = weights_desc.to_format_any();
  auto bias_desc_query =
      b.has_value() ? bias_desc.to_format_any() : tensor::desc();
  auto dst_desc_query = dst_desc.to_format_any();

  // For nhwc path, weight uses format_tag::any,
  // while activation uses format_tag::nhwc.
  bool is_nhwc = src_desc.is_nhwc() || weights_desc.is_nhwc();
  if (is_nhwc) {
    src_desc_query = src_desc.to_format(tag::nhwc);
    weights_desc_query = weights_desc.to_format_any();
    bias_desc_query =
        b.has_value() ? bias_desc.to_format_any() : tensor::desc();
    dst_desc_query = dst_desc.to_format(tag::nhwc);
  }

  convolution_forward::primitive_desc pd;
  if (b.has_value()) {
    pd = convolution_forward::primitive_desc(
        {prop_kind::forward,
         algorithm::convolution_direct,
         src_desc_query,
         weights_desc_query,
         bias_desc_query,
         dst_desc_query,
         {stride.begin(), stride.end()},
         dilates,
         {padding.begin(), padding.end()},
         {padding.begin(), padding.end()}},
        op_attr,
        aengine);
  } else {
    pd = convolution_forward::primitive_desc(
        {prop_kind::forward,
         algorithm::convolution_direct,
         src_desc_query,
         weights_desc_query,
         dst_desc_query,
         {stride.begin(), stride.end()},
         dilates,
         {padding.begin(), padding.end()},
         {padding.begin(), padding.end()}},
        op_attr,
        aengine);
  }

  // allocate scratchpad
  tensor scratchpad(pd.scratchpad_desc());
  auto expected_src = x.reorder_if_differ_in(pd.src_desc());
  auto expected_weights =
      w.make_grouped_weights(groups).reorder_if_differ_in(pd.weights_desc());
  y.reinit_if_possible(pd.dst_desc());
  // convolution_forward conv_prim(pd);
  if (b.has_value()) {
    convolution_forward(pd).execute(
        utils::stream::default_stream(),
        {{ZENDNN_ARG_SRC, expected_src},
         {ZENDNN_ARG_WEIGHTS, expected_weights},
         {ZENDNN_ARG_BIAS, b.value()},
         {ZENDNN_ARG_DST, y},
         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  } else {
    convolution_forward(pd).execute(
        utils::stream::default_stream(),
        {{ZENDNN_ARG_SRC, expected_src},
         {ZENDNN_ARG_WEIGHTS, expected_weights},
         {ZENDNN_ARG_DST, y},
         {ZENDNN_ARG_SCRATCHPAD, scratchpad}});
  }
  return y;
}
} // namespace zendnn

namespace at {
namespace native {

Tensor zendnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      !(input.scalar_type() != ScalarType::Float),
      "zendnn_convolution: Incorrect data type of input tensor");
  TORCH_CHECK(
      !(weight.scalar_type() != ScalarType::Float),
      "zendnn_convolution: Incorrect data type of weights tensor");

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::optional<zendnn::tensor> zendnn_bias{c10::nullopt};
  if (bias.defined()) {
    TORCH_CHECK(
        !(bias.scalar_type() != ScalarType::Float),
        "zendnn_convolution: Incorrect data type in bias");
    zendnn_bias = itensor_from_tensor(bias);
  }
  const zendnn::tensor zendnn_input = itensor_from_tensor(input);
  const zendnn::tensor zendnn_weight = itensor_from_tensor(weight);

  zendnn::tensor zendnn_output = _zendnn_convolution(
      zendnn_input,
      zendnn_weight,
      zendnn_bias,
      padding,
      stride,
      dilation,
      groups);

  if (input.is_zendnn()) {
    return new_with_itensor_zendnn(
        std::move(zendnn_output),
        optTypeMetaToScalarType(input.options().dtype_opt()),
        input.options().device_opt());
  } else {
    return zendnn_to_dense(new_with_itensor_zendnn(
        std::move(zendnn_output),
        optTypeMetaToScalarType(input.options().dtype_opt()),
        input.options().device_opt()));
  }
}

Tensor zendnn_convolution_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward_input: backward propegation is disabled in zendnn");
}

std::tuple<Tensor, Tensor> zendnn_convolution_backward_weights(
    IntArrayRef weight_size,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward_weights: backward propegation is disabled in zendnn");
}

std::tuple<Tensor, Tensor, Tensor> zendnn_convolution_backward(
    const Tensor& input,
    const Tensor& grad_output_t,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      false,
      "zendnn_convolution_backward_weights: backward propegation is disabled in zendnn");
}

REGISTER_ALL_CPU_DISPATCH(
    zendnn_convolution_backward_stub,
    &zendnn_convolution_backward);

} // namespace native
} // namespace at

#endif
