#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>

namespace at { namespace native {

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

#if AT_MKLDNN_ENABLED()

#define ATTR_FUNC(NAME)                              \
  [](torch::List<c10::optional<at::Scalar>> scalars, \
     c10::optional<c10::string_view> algorithm) {    \
    return ideep::attr_t::fuse_##NAME();             \
  }

AttrFunction attr_func_leaky_relu =
    [](torch::List<c10::optional<at::Scalar>> scalars,
       c10::optional<c10::string_view> algorithm) {
      TORCH_CHECK(
          scalars.size() == 1 &&
              scalars[0].get().toOptional<at::Scalar>().has_value(),
          "leaky_relu is expected to have one scalar input: negative_slope");
      auto alpha_value =
          scalars[0].get().toOptional<at::Scalar>().value().to<float>();
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };

AttrFunction attr_func_hardtanh =
    [](torch::List<c10::optional<at::Scalar>> scalars,
       c10::optional<c10::string_view> algorithm) {
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

AttrFunction attr_func_gelu = [](torch::List<c10::optional<at::Scalar>> scalars,
                                 c10::optional<c10::string_view> algorithm) {
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

const std::map<c10::string_view, AttrFunction>& fx_fusion_attr_map() {
  static const std::map<c10::string_view, AttrFunction> fusion_attr_map{
      {"relu", ATTR_FUNC(relu)},
      {"sigmoid", ATTR_FUNC(sigmoid)},
      {"tanh", ATTR_FUNC(tanh)},
      {"hardswish", ATTR_FUNC(hardswish)},
      {"leaky_relu", attr_func_leaky_relu},
      {"hardtanh", attr_func_hardtanh},
      {"gelu", attr_func_gelu},
  };
  return fusion_attr_map;
};

#endif // AT_MKLDNN_ENABLED()
}}
