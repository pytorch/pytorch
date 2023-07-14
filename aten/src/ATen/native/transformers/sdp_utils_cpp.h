#pragma once
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymFloat.h>
#include <cmath>
#include <cstdint>
namespace sdp {

constexpr int32_t num_backends = 3;
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

// Note that if this changed make sure to update
// the templated enum in mem_eff/kernel_forward.h and mem_eff/kernel_backward.h
enum class CustomMaskType {
  NoCustomMask = 0,
  CausalFromTopLeft = 1,
  CausalFromBottomRight = 2,
  NumCustomMaskTypes,
};

struct sdp_params {
  const at::Tensor& query;
  const at::Tensor& key;
  const at::Tensor& value;
  bool has_attn_mask;
  double dropout;
  bool is_causal;
};

SDPBackend select_sdp_backend_cpp(sdp_params kernel_params);

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

} // namespace sdp
