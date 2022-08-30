#include <vector>

#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/LinearPrepack.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace linear {

c10::intrusive_ptr<mkldnn::LinearOpContext> createLinearPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    std::vector<int64_t> input_size,
    std::string attr,
    std::vector<c10::optional<at::Scalar>> scalars,
    c10::optional<std::string> algorithm) {
  auto it = fusion_attr_map().find(attr);
  TORCH_CHECK(it != fusion_attr_map().end(), "Fusion behavior undefined.");
  ideep::attr_t op_attr = it->second(scalars, algorithm);
  return mkldnn::MkldnnLinearOpContext::create_context(
      std::move(weight), std::move(bias), std::move(input_size), op_attr);
}

ContextLinear create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef input_size,
    const ideep::attr_t& attr) {
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor w = itensor_view_from_dense(weight);
  auto dtype = w.get_data_type();

  int64_t b_size = std::accumulate(
                       input_size.begin(),
                       input_size.end(),
                       (int64_t)1,
                       std::multiplies<int64_t>()) /
      input_size[input_size.size() - 1];

  auto out_features = weight.size(0);
  auto in_features = weight.size(1);
  ideep::dims reshaped_input_size = {b_size, in_features};

  ideep::tensor::desc expected_weight_desc =
      ideep::inner_product_forward::expected_weights_desc(
          {out_features, in_features},
          reshaped_input_size,
          /* w_dtype */ dtype,
          /* x_dtype */ dtype);

  ideep::tensor packed_weight;
  packed_weight.init(expected_weight_desc);
  packed_weight.feed_from(w);

  return ContextLinear{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      std::move(attr)};
}

void _mkldnn_linear_out(
    const ideep::tensor& x,
    ideep::tensor& y,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    const ideep::attr_t& attr = ideep::attr_t()) {
  if (b.has_value()) {
    ideep::inner_product_forward::compute(
        x,
        w,
        b.value(),
        y,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        attr);
  } else {
    ideep::inner_product_forward::compute(
        x, w, y, ideep::scale_t(), ideep::scale_t(), ideep::scale_t(), attr);
  }
}

void mkldnn_linear_out(
    const Tensor& input,
    ideep::tensor& mkldnn_output,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<Tensor>& bias_opt,
    const ideep::attr_t& attr = ideep::attr_t()) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input);

  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  _mkldnn_linear_out(
      mkldnn_input, mkldnn_output, mkldnn_weight, mkldnn_bias, attr);
}

Tensor run(ContextLinear& context, const Tensor& input) {
  const ideep::tensor& mkldnn_weight = context.weight_packed_;

  auto input_size = input.sizes();

  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(mkldnn_weight.get_dim(0));
  auto output = at::empty(output_size, input.options());

  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {input_reshaped.size(0),
                                                 mkldnn_weight.get_dim(0)};
    output = output.reshape(output_size_reshaped);
  }

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  mkldnn_linear_out(
      input_reshaped,
      mkldnn_output,
      mkldnn_weight,
      context.at_bias_,
      context.attr_);

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

void run(ContextLinear& context, const Tensor& input, void* output) {
  const ideep::tensor& mkldnn_weight = context.weight_packed_;

  auto input_size = input.sizes();

  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(mkldnn_weight.get_dim(0));

  std::vector<int64_t> output_size_reshaped = {input_reshaped.size(0),
                                               mkldnn_weight.get_dim(0)};

  ideep::tensor::desc o_desc = {output_size_reshaped,
                                get_mkldnn_dtype(input.scalar_type())};
  ideep::tensor mkldnn_output = {o_desc, output};

  mkldnn_linear_out(
      input_reshaped,
      mkldnn_output,
      mkldnn_weight,
      context.at_bias_,
      context.attr_);
}

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::LinearOpContext>& op_context) {
  return op_context->run(input);
}

} // namespace linear
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
