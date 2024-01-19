
#include "FusionUtils.h"

using namespace at::native::xpu::onednn;

namespace at {
namespace native::xpu {

#define ATTR_FUNC(NAME)                                                       \
  [](xpu::onednn::Attr attr) {                                                \
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_##NAME); \
  }

xpu::onednn::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars =
        torch::List<c10::optional<at::Scalar>>(),
    c10::optional<c10::string_view> algorithm = c10::nullopt,
    xpu::onednn::Attr attr = Attr()) {
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

xpu::onednn::Attr string_to_unary_attr(
    c10::string_view unary,
    xpu::onednn::Attr attr) {
  if (unary == "relu") {
    return ATTR_FUNC(relu)(attr);
  } else if (unary == "sigmoid") {
    return ATTR_FUNC(sigmoid)(attr);
  } else if (unary == "tanh") {
    return ATTR_FUNC(tanh)(attr);
  } else if (unary == "hardswish") {
    return unary_attr_with_arg(
        "hardswish",
        torch::List<c10::optional<at::Scalar>>(),
        c10::nullopt,
        attr);
  } else if (unary == "swish") {
    return unary_attr_with_arg(
        "silu", torch::List<c10::optional<at::Scalar>>(), c10::nullopt, attr);
  }
  return attr;
}

xpu::onednn::Attr construct_unary_attr(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    Attr attr = Attr()) {
  std::set<c10::string_view> simple_unary = {
      "relu", "sigmoid", "tanh", "hardswish", "swish"};
  if (simple_unary.find(unary) != simple_unary.end()) {
    return string_to_unary_attr(unary, attr);
  } else {
    return unary_attr_with_arg(unary, scalars, algorithm, attr);
  }
}

xpu::onednn::Attr construct_binary_attr(
    c10::string_view binary,
    c10::optional<at::Scalar> alpha,
    const Tensor& other,
    xpu::onednn::Attr attr = Attr()) {
  if (binary == "mul") {
    attr.append_post_binary(attr.kind_with_binary_mul, other);
  } else if (binary == "sub") {
    attr.append_post_binary(attr.kind_with_binary_sub, other);
  } else if (binary == "div") {
    attr.append_post_binary(attr.kind_with_binary_div, other);
  } else if (binary == "add") {
    attr.append_post_binary(attr.kind_with_binary_add, other);
  }
  return attr;
}

} // namespace native::xpu
} // namespace at
