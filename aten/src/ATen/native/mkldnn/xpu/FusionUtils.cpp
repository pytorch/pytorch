#include <ATen/native/mkldnn/xpu/FusionUtils.h>

using namespace at::native::onednn;

namespace at::native::xpu {

onednn::Attr& handle_argument_less(std::string_view unary, onednn::Attr& attr) {
  static const std::unordered_map<
      std::string_view,
      std::function<onednn::Attr&(onednn::Attr&)>>
      unary_map = {
          {"relu",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 0.0f, 0.0f, attr.kind_with_relu);
           }},
          {"sigmoid",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 0.0f, 0.0f, attr.kind_with_sigmoid);
           }},
          {"tanh",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 0.0f, 0.0f, attr.kind_with_tanh);
           }},
          {"hardswish",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 1.0f / 6.0f, 1.0f / 2.0f, attr.kind_with_hardswish);
           }},
          {"swish",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 1.0f, 0.0f, attr.kind_with_swish);
           }},
          {"hardsigmoid",
           [](onednn::Attr& attr) -> onednn::Attr& {
             return attr.append_post_eltwise(
                 1.0f, 1.0f / 6.0f, 1.0f / 2.0f, attr.kind_with_hardsigmoid);
           }},
          {"none", [](onednn::Attr& attr) -> onednn::Attr& { return attr; }}};

  if (unary_map.find(unary) != unary_map.end()) {
    return unary_map.at(unary)(attr);
  }
  TORCH_CHECK(
      false,
      "Unary attr ",
      unary,
      " is not supported for conv/linear post unary fusion");
}

onednn::Attr& handle_need_sclars(
    std::string_view unary,
    onednn::Attr& attr,
    torch::List<std::optional<at::Scalar>> scalars) {
  static const std::unordered_map<
      std::string_view,
      std::function<onednn::Attr&(
          onednn::Attr&, torch::List<std::optional<at::Scalar>>)>>
      unary_map = {
          {"leaky_relu",
           [](onednn::Attr& attr,
              torch::List<std::optional<at::Scalar>> scalars) -> onednn::Attr& {
             auto alpha =
                 scalars[0].get().toOptional<at::Scalar>().value().to<float>();
             return attr.append_post_eltwise(
                 1.0f, alpha, 0.f, attr.kind_with_relu);
           }},
          {"hardtanh",
           [](onednn::Attr& attr,
              torch::List<std::optional<at::Scalar>> scalars) -> onednn::Attr& {
             auto alpha =
                 scalars[0].get().toOptional<at::Scalar>().value().to<float>();
             auto beta =
                 scalars[1].get().toOptional<at::Scalar>().value().to<float>();
             return attr.append_post_eltwise(
                 1.0f, alpha, beta, attr.kind_with_clip);
           }}};

  if (unary_map.find(unary) != unary_map.end()) {
    return unary_map.at(unary)(attr, scalars);
  }
  TORCH_CHECK(
      false,
      "Unary attr ",
      unary,
      " is not supported for conv/linear post unary fusion");
}

onednn::Attr& handle_need_algorithm(
    std::string_view unary,
    onednn::Attr& attr,
    std::optional<std::string_view> algorithm) {
  TORCH_CHECK(
      unary == "gelu",
      "GELU is the only unary operation that requires an algorithm currently");
  if (!algorithm.has_value()) {
    TORCH_CHECK(
        false,
        "GELU algorithm is not specified, please specify it as 'none' or 'tanh'");
  }
  enum dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = attr.kind_with_gelu_erf;
  } else {
    gelu_type = attr.kind_with_gelu_tanh;
  }
  return attr.append_post_eltwise(1.0f, 0.0f, 0.0f, gelu_type);
}

onednn::Attr& construct_unary_attr(
    onednn::Attr& attr,
    std::string_view unary,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  // Define sets for unary operations based on their argument requirements.
  // Category `argument_less`: stateless operations
  // Category `need_scalars`: require alpha/beta
  // Category `need_algorithm`: require algorithm specification, only gelu now.
  // If further unary operations required, they can be added to these sets or
  // add new sets according to their new categories.
  static const std::set<std::string_view> argument_less = {
      "relu", "sigmoid", "tanh", "hardswish", "swish", "hardsigmoid"};
  static const std::set<std::string_view> need_scalars = {
      "leaky_relu", "hardtanh"};
  static const std::set<std::string_view> need_algorithm = {"gelu"};

  if (argument_less.find(unary) != argument_less.end()) {
    return handle_argument_less(unary, attr);
  } else if (need_scalars.find(unary) != need_scalars.end()) {
    return handle_need_sclars(unary, attr, scalars);
  } else if (need_algorithm.find(unary) != need_algorithm.end()) {
    return handle_need_algorithm(unary, attr, algorithm);
  } else {
    TORCH_CHECK(
        false,
        "Unary attr ",
        unary,
        " is not supported for conv/linear post unary fusion");
  }
}

} // namespace at::native::xpu
