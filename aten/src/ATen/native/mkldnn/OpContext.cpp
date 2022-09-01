#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

#define ATTR_FUNC(NAME)                              \
  [](std::vector<c10::optional<at::Scalar>> scalars, \
     c10::optional<std::string> algorithm) {         \
    return ideep::attr_t::fuse_##NAME();             \
  }

AttrFunction attr_func_none = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  const static ideep::attr_t empty_attr = ideep::attr_t();
  return empty_attr;
};

AttrFunction attr_func_sum = [](std::vector<c10::optional<at::Scalar>> scalars,
                                c10::optional<std::string> algorithm) {
  float alpha_value = scalars[0] ? scalars[0].value().to<float>() : 1.0;
  return ideep::attr_t::fuse_sum(alpha_value);
};

AttrFunction attr_func_sum_relu =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      float alpha_value = scalars[0] ? scalars[0].value().to<float>() : 1.0;
      return ideep::attr_t::residual(alpha_value);
    };

const std::map<std::string, AttrFunction>& fusion_attr_map() {
  static const std::map<std::string, AttrFunction> fusion_attr_map{
      {"none", attr_func_none},
      {"relu", ATTR_FUNC(relu)},
      {"sum", attr_func_sum},
      {"sum_relu", attr_func_sum_relu},
  };
  return fusion_attr_map;
};

c10::intrusive_ptr<ConvOpContext> MkldnnConvOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  auto op_context = mkldnn::internal::convolution::create(
      weight, bias, padding, stride, dilation, groups, input_size, attr);

  auto conv_op_context = c10::make_intrusive<MkldnnConvOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      std::move(op_context));

  return conv_op_context;
}

Tensor MkldnnConvOpContext::run(const Tensor& input) {
  return mkldnn::internal::convolution::run(op_context_, input);
}

Tensor& MkldnnConvOpContext::sum_run(const Tensor& input, Tensor& other) {
  return mkldnn::internal::convolution::sum_run(op_context_, input, other);
}

void MkldnnConvOpContext::run(const Tensor& input, void* output) {
  return mkldnn::internal::convolution::run(op_context_, input, output);
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
