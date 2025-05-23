#include <ATen/native/mkldnn/xpu/FusionUtils.h>

using namespace at::native::onednn;

namespace at::native::xpu {

onednn::Attr& unary_attr_with_arg(
    onednn::Attr& attr,
    std::string_view unary,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  if (unary == "hardswish") {
    return attr.append_post_eltwise(
        1.0f, 1.f / 6.f, 1.f / 2.f, attr.kind_with_hardswish);
  } else if (unary == "silu") {
    return attr.append_post_eltwise(1.0f, 1.0f, 0.0f, attr.kind_with_swish);
  } else if (unary == "leaky_relu") {
    auto alpha = scalars[0].get().toOptional<at::Scalar>().value().to<float>();
    return attr.append_post_eltwise(1.0, alpha, 0.f, attr.kind_with_relu);
  } else if (unary == "hardtanh") {
    auto alpha = scalars[0].get().toOptional<at::Scalar>().value().to<float>();
    auto beta = scalars[1].get().toOptional<at::Scalar>().value().to<float>();
    return attr.append_post_eltwise(1.0f, alpha, beta, attr.kind_with_clip);
  } else if (unary == "gelu") {
    TORCH_CHECK(algorithm.has_value(), "GELU algorithm is not specified");
    enum dnnl::algorithm gelu_type;
    if (algorithm.value() == "none") {
      gelu_type = attr.kind_with_gelu_erf;
    } else {
      gelu_type = attr.kind_with_gelu_tanh;
    }
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, gelu_type);
  } else if (unary == "hardsigmoid") {
    return attr.append_post_eltwise(
        1.0f, 1.0f / 6.f, 1.0f / 2.0f, attr.kind_with_hardsigmoid);
  }
  TORCH_CHECK(
      unary == "none",
      "Unary attr ",
      unary,
      "is not supported for conv/linear post unary fusion");
  return attr;
}

onednn::Attr& string_to_unary_attr(onednn::Attr& attr, std::string_view unary) {
  if (unary == "relu") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_relu);
  } else if (unary == "sigmoid") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_sigmoid);
  } else if (unary == "tanh") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_tanh);
  } else if (unary == "hardswish") {
    return unary_attr_with_arg(
        attr,
        "hardswish",
        torch::List<std::optional<at::Scalar>>(),
        std::nullopt);
  } else if (unary == "swish") {
    return unary_attr_with_arg(
        attr, "silu", torch::List<std::optional<at::Scalar>>(), std::nullopt);
  }
  return attr;
}

onednn::Attr& construct_unary_attr(
    onednn::Attr& attr,
    std::string_view unary,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  static const std::set<std::string_view> simple_unary = {
      "relu", "sigmoid", "tanh", "hardswish", "swish"};
  if (simple_unary.find(unary) != simple_unary.end()) {
    return string_to_unary_attr(attr, unary);
  } else {
    return unary_attr_with_arg(attr, unary, scalars, algorithm);
  }
}

} // namespace at::native::xpu
