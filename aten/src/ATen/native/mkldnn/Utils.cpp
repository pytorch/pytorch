#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>

namespace at::native {

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());
  // copy N and C
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  for (const auto i : c10::irange(2, input_size.size())) {
    output_size[i] = pooling_output_shape_pad_lr<int64_t>(
      input_size[i],
      kernel_size[i - 2],
      padding_l[i - 2],
      padding_r[i - 2],
      stride[i - 2],
      dilation[i - 2],
      ceil_mode
    );
  }

   return output_size;
}

void check_mkldnn_binary_fusion_inputs(
    const Tensor& input,
    const Tensor& other,
    const Tensor& weight,
    const Tensor& bias) {
  if (!weight.is_mkldnn()) {
    TORCH_CHECK(
        input.options().type_equal(weight.options()),
        "Input type (",
        input.toString(),
        ") and weight type (",
        weight.toString(),
        ") should be the same");
  } else {
    TORCH_CHECK(
        input.scalar_type() == input.scalar_type(),
        "mkldnn pointwise binary: input dtype and weight dtype should be the same");
  }
  TORCH_CHECK(
      input.options().type_equal(other.options()),
      "Input type (",
      input.toString(),
      ") and other type (",
      other.toString(),
      ") should be the same");
  TORCH_CHECK(
      !bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (",
      input.toString(),
      ") and bias type (",
      bias.toString(),
      ") should be the same");
  TORCH_CHECK(
      input.device().is_cpu(),
      "mkldnn pointwise binary fusion: input's device should be CPU");
  TORCH_CHECK(
      input.scalar_type() == ScalarType::Float ||
          input.scalar_type() == ScalarType::BFloat16 ||
          input.scalar_type() == ScalarType::Half,
      "mkldnn pointwise binary: input's dtype should be float, bfloat16 or half");
  mkldnn_check_low_precision(input.scalar_type(), "mkldnn pointwise binary");
}

#if AT_MKLDNN_ENABLED()

#define ATTR_FUNC(NAME)                              \
  [](torch::List<std::optional<at::Scalar>> scalars, \
     std::optional<std::string_view> algorithm) {    \
    return ideep::attr_t::fuse_##NAME();             \
  }

AttrFunction attr_func_leaky_relu =
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<std::string_view> algorithm) {
      TORCH_CHECK(
          scalars.size() == 1 &&
              scalars[0].get().toOptional<at::Scalar>().has_value(),
          "leaky_relu is expected to have one scalar input: negative_slope");
      auto alpha_value =
          scalars[0].get().toOptional<at::Scalar>().value().to<float>();
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };

AttrFunction attr_func_hardtanh =
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<std::string_view> algorithm) {
      TORCH_CHECK(
          scalars.size() == 2 &&
              scalars[0].get().toOptional<at::Scalar>().has_value() &&
              scalars[1].get().toOptional<at::Scalar>().has_value(),
          "hardtanh is expected to have two scalar input: min_val and max_val");

      auto lower_bound_value =
          scalars[0].get().toOptional<at::Scalar>().value().to<float>();
      auto upper_bound_value =
          scalars[1].get().toOptional<at::Scalar>().value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

AttrFunction attr_func_gelu = [](torch::List<std::optional<at::Scalar>> scalars,
                                 std::optional<std::string_view> algorithm) {
  TORCH_CHECK(
      algorithm.has_value(),
      "gelu is expected to have one str input: algorithm");
  dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (algorithm.value() == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported gelu algorithm: ", algorithm.value());
  }

  return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
};

AttrFunction attr_func_hardsigmoid =
    [](torch::List<std::optional<at::Scalar>> scalars,
       std::optional<std::string_view> algorithm) {
      ideep::attr_t attr;
      ideep::post_ops po;
      po.append_eltwise(
          ideep::algorithm::eltwise_hardsigmoid, 1.0f / 6.0f, 0.5f);
      attr.set_post_ops(po);
      return attr;
    };

const std::map<std::string_view, AttrFunction>& fusion_unary_attr_map() {
  static const std::map<std::string_view, AttrFunction> fusion_attr_map{
      {"relu", ATTR_FUNC(relu)},
      {"sigmoid", ATTR_FUNC(sigmoid)},
      {"tanh", ATTR_FUNC(tanh)},
      {"swish", ATTR_FUNC(swish)},
      {"hardswish", ATTR_FUNC(hardswish)},
      {"hardsigmoid", attr_func_hardsigmoid},
      {"leaky_relu", attr_func_leaky_relu},
      {"hardtanh", attr_func_hardtanh},
      {"gelu", attr_func_gelu},
  };
  return fusion_attr_map;
}

const std::map<std::string_view, ideep::algorithm>& fusion_unary_alg_map() {
  static const std::map<std::string_view, ideep::algorithm> fusion_attr_map{
      {"relu", {ideep::algorithm::eltwise_relu}},
  };
  return fusion_attr_map;
}

const std::map<std::string_view, ideep::algorithm>& fusion_binary_alg_map() {
  static const std::map<std::string_view, ideep::algorithm> fusion_attr_map{
      {"add", {ideep::algorithm::binary_add}},
      {"sub", {ideep::algorithm::binary_sub}},
      {"mul", {ideep::algorithm::binary_mul}},
      {"div", {ideep::algorithm::binary_div}},
  };
  return fusion_attr_map;
}

#endif // AT_MKLDNN_ENABLED()
}
