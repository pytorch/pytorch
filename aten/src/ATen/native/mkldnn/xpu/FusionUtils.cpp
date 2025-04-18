
#include <ATen/native/mkldnn/xpu/FusionUtils.h>

using namespace at::native::onednn;

namespace at::native::xpu {

onednn::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars =
        torch::List<c10::optional<at::Scalar>>(),
    c10::optional<c10::string_view> algorithm = c10::nullopt,
    onednn::Attr attr = Attr()) {
  if (unary == "hardswish") {
    return attr.append_post_eltwise(
        1.0f, 1.f / 6.f, 1.f / 2.f, attr.kind_with_hardswish);
  } else if (unary == "silu") {
    return attr.append_post_eltwise(1.0f, 1.0f, 0.0f, attr.kind_with_swish);
  } else if (unary == "leaky_relu") {
    auto alpha = scalars[0].get().toOptional<at::Scalar>().value().to<float>();
    return attr.append_post_eltwise(1.0, alpha, 0.f, attr.kind_with_relu);
  } else if (unary == "clamp") {
    auto alpha = scalars[0].get().toOptional<at::Scalar>().value().to<float>();
    auto beta = scalars[1].get().toOptional<at::Scalar>().value().to<float>();
    return attr.append_post_eltwise(1.0f, alpha, beta, attr.kind_with_clip);
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
  } else if(unary == "abs") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_abs);
  } else if(unary == "pow") {
    auto beta = scalars[1].get().toOptional<at::Scalar>().value().to<float>();
    return attr.append_post_eltwise(1.0f, 1.0f, beta, attr.kind_with_pow); 
  }else if (unary == "hardsigmoid") {
    return attr.append_post_eltwise(
        1.0f, 1.0f / 6.f, 1.0f / 2.0f, attr.kind_with_hardsigmoid);
  }
  TORCH_CHECK(
      unary == "none",
      "Unary attr ",
      unary,
      " is not supported for conv/linear post unary fusion");
  return attr;
}

onednn::Attr string_to_unary_attr(c10::string_view unary, onednn::Attr attr) {
  if (unary == "relu") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_relu);
  } else if (unary == "sigmoid") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_sigmoid);
  } else if (unary == "tanh") {
    return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_tanh);
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

onednn::Attr construct_unary_attr(
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



} // namespace at::native::xpu
