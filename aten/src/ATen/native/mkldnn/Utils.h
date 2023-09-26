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

static inline std::vector<int64_t> padding_r(
    IntArrayRef padding, IntArrayRef output_padding)
{
  // ConvTranpose padding adjustment
  //
  // PyTorch uses padding/output_padding:
  //   osize = (isize - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
  //
  // MKLDNN uses padding_l/padding_r:
  //   osize = (isize - 1) * stride - padding_l - padding_r + dilation * (kernel_size - 1) + 1
  //
  // So: padding_l = padding, padding_r = padding - output_padding
  //
  auto dim = padding.size();
  std::vector<int64_t> pad_r(dim);
  for (const auto d : c10::irange(dim)) {
    pad_r[d] = padding[d] - output_padding[d];
  }
  return pad_r;
}

#if AT_MKLDNN_ENABLED()

using AttrFunction = std::function<ideep::attr_t(
    torch::List<c10::optional<at::Scalar>>,
    c10::optional<c10::string_view>)>;

const std::map<c10::string_view, AttrFunction>& fusion_unary_attr_map();

const std::map<c10::string_view, ideep::algorithm>& fusion_unary_alg_map();

const std::map<c10::string_view, ideep::algorithm>& fusion_binary_alg_map();

#endif // AT_MKLDNN_ENABLED()
};

#if AT_MKLDNN_ENABLED()
inline bool mkldnn_bf16_device_check() {
  return ideep::has_bf16_type_support() || (cpuinfo_initialize() && cpuinfo_has_arm_bf16());
}
inline bool mkldnn_fp16_device_check() {
  return ideep::has_fp16_type_support();
}
#else
inline bool mkldnn_bf16_device_check() {
  return false;
}
inline bool mkldnn_fp16_device_check() {
  return false;
}
#endif

inline void mkldnn_check_low_precision(ScalarType input_t, std::string name) {
  if (input_t == ScalarType::BFloat16) {
    TORCH_CHECK(
        mkldnn_bf16_device_check(),
        name,
        ": bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq");
  } else if (input_t == ScalarType::Half) {
    TORCH_CHECK(
        mkldnn_fp16_device_check(),
        name,
        ": fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
  }
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
