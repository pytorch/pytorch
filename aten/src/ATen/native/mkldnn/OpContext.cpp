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

static constexpr float kMin = -std::numeric_limits<float>::infinity();
static constexpr float kMax = std::numeric_limits<float>::infinity();

AttrFunction attr_func_none = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  const static ideep::attr_t empty_attr = ideep::attr_t();
  return empty_attr;
};

AttrFunction attr_func_leaky_relu =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto alpha_value = scalars[0].value().to<float>();
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };

AttrFunction attr_func_hardtanh =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto lower_bound_value = scalars[0].value().to<float>();
      auto upper_bound_value = scalars[1].value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

AttrFunction attr_func_gelu = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (algorithm.value() == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false, "ipex::linear_gelu_run only support tanh approximate now");
  }

  return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
};

AttrFunction attr_func_clamp =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      float lower_bound_value =
          scalars[0] ? scalars[0].value().to<float>() : kMin;
      float upper_bound_value =
          scalars[1] ? scalars[1].value().to<float>() : kMax;

      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

const std::map<std::string, AttrFunction>& fusion_attr_map() {
  static const std::map<std::string, AttrFunction> fusion_attr_map{
      {"none", attr_func_none},
      {"relu", ATTR_FUNC(relu)},
      {"sigmoid", ATTR_FUNC(sigmoid)},
      {"tanh", ATTR_FUNC(tanh)},
      {"leaky_relu", attr_func_leaky_relu},
      {"hardtanh", attr_func_hardtanh},
      {"gelu", attr_func_gelu},
      {"clamp", attr_func_clamp},
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

void MkldnnConvOpContext::run(const Tensor& input, void* output) {
  return mkldnn::internal::convolution::run(op_context_, input, output);
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
