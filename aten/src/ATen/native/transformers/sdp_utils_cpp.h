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

// This helper function creates a constexpr std::array
// From a compile time list of values
template <typename V, typename... T>
constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  return {{std::forward<T>(t)...}};
}

bool input_requires_grad(sdp_params params);

bool has_for_nested_inputs(sdp_params params);

std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params params);

template <typename dtype_vector>
bool check_tensor_dtype(
    sdp_params params,
    dtype_vector allowed_dtypes,
    bool debug);

bool try_broadcast_param_size(
    const c10::SymInt q_size,
    const c10::SymInt k_size,
    const c10::SymInt v_size,
    c10::string_view param_name,
    bool debug);

bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
    at::Tensor param,
    c10::string_view param_name,
    bool debug);

bool check_for_seq_len_0_nested_tensor(sdp_params params, bool debug);

bool check_nested_tensor(sdp_params params, bool debug);

bool check_for_dropout(sdp_params params, bool debug);

bool check_requires_grad_and_nested(sdp_params params, bool debug);

bool check_for_attn_mask(sdp_params params, bool debug);

bool check_for_noncontiguous(sdp_params params, bool debug);

bool check_tensor_shapes(sdp_params params, bool debug);

bool check_safe_kv_broadcast(at::Tensor param, bool debug);

bool check_batch_size_and_num_heads_cpp(sdp_params params, bool debug);

bool check_head_dim_size_cpp(sdp_params params, bool debug);

bool check_head_dim_size_mem_efficient_cpp(sdp_params params, bool debug);

bool check_runtime_disabled_flash(sdp_params params, bool debug);

bool check_runtime_disabled_mem_efficient(sdp_params params, bool debug);

bool use_flash_attention_cpp(sdp_params params, bool debug);

bool use_mem_efficient_attention_cpp(sdp_params params, bool debug);
} // namespace sdp
