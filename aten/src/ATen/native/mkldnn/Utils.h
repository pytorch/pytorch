#pragma once

#include <ATen/Config.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <cpuinfo.h>
#include <vector>

#if AT_MKLDNN_ENABLED()
#include <ideep/tensor.hpp>
#endif // AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace = false);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);

void check_mkldnn_binary_fusion_inputs(
    const Tensor& input,
    const Tensor& other,
    const Tensor& weight,
    const Tensor& bias);

#if AT_MKLDNN_ENABLED()

using AttrFunction = std::function<ideep::attr_t(
    torch::List<c10::optional<at::Scalar>>,
    c10::optional<c10::string_view>)>;

const std::map<c10::string_view, AttrFunction>& fusion_unary_attr_map();

const std::map<c10::string_view, ideep::algorithm>& fusion_unary_alg_map();

const std::map<c10::string_view, ideep::algorithm>& fusion_binary_alg_map();

#endif // AT_MKLDNN_ENABLED()
};

inline bool mkldnn_bf16_device_check() {
  return cpuinfo_initialize() && ((cpuinfo_has_x86_avx512bw()
     && cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512dq()) || (cpuinfo_has_arm_bf16()));
}

#if defined(__aarch64__)
inline bool mkldnn_bf16_device_check_arm() {
  return (cpuinfo_initialize() && cpuinfo_has_arm_bf16());
}
#else
constexpr bool mkldnn_bf16_device_check_arm() {
  return false;
}
#endif

}
