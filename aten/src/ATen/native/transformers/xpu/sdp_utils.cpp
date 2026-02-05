#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/native/transformers/xpu/sdp_utils.h>
#include <c10/util/Array.h>

namespace sdp {

bool is_flash_attention_available() {
  return sycltla::is_flash_attention_available();
}

inline bool is_flash_attention_available(sdp_params const& params, bool debug) {
  if (!is_flash_attention_available()) {
    if (debug) {
      TORCH_WARN("Torch XPU was not compiled with flash attention.");
    }
    return false;
  }
  return true;
}

bool check_flash_attention_hardware_support(
    sdp_params const& params,
    bool debug) {
  if (!at::xpu::is_available()) {
    TORCH_CHECK(false, "FlashAttentionXPU: XPU device is not available.");
  }

  constexpr auto supported_architectures =
      c10::array_of<sycl::ext::oneapi::experimental::architecture>(
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21);
  auto* device_prop = at::xpu::getCurrentDeviceProperties();
  auto device_architecture = device_prop->architecture;

  if (std::find(
          supported_architectures.begin(),
          supported_architectures.end(),
          device_architecture) == supported_architectures.end()) {
    if (debug) {
      TORCH_WARN(
          "XPU device architecture does not support flash attention. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21.");
    }
    return false;
  }

  return true;
}

inline bool check_flash_attention_datatype(
    sdp_params const& params,
    bool debug) {
  constexpr auto supported_dtypes =
      c10::array_of<at::ScalarType>(at::kBFloat16, at::kHalf);

  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(
             supported_dtypes.begin(), supported_dtypes.end(), query_dtype) !=
         supported_dtypes.end()))) {
    if (debug) {
      TORCH_WARN(
          "FlashAttentionXPU expected query, key, and value to all be of dtype: {",
          "bfloat16, half",
          "}. Got ",
          "Query dtype: ",
          params.query.dtype(),
          ", Key dtype: ",
          params.key.dtype(),
          ", and Value dtype: ",
          params.value.dtype(),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_flash_attention_head_dim_size(
    sdp_params const& params,
    bool debug) {
  const int query_size_last = params.query.size(3);
  const int key_size_last = params.key.size(3);
  const int value_size_last = params.value.size(3);

  const bool head_dims_equal = (query_size_last == key_size_last) &&
      (query_size_last == value_size_last);
  if (!head_dims_equal) {
    if (debug) {
      TORCH_WARN(
          "FlashAttentionXPU requires q,k,v to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }

  constexpr auto max_supported_headdim = 192;
  if (query_size_last > max_supported_headdim) {
    if (debug) {
      TORCH_WARN(
          "FlashAttentionXPU supports head dimension up to ",
          max_supported_headdim,
          ". ",
          "Got head dimension: ",
          query_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_flash_attention_layout(sdp_params const& params, bool debug) {
  return sycltla::check_flash_attention_layout(params, debug);
}

inline bool check_flash_causal_non_square_seqlens(
    sdp_params const& params,
    bool debug) {
  // FlashAttention 2 updated the default mask meaning for causal in this PR:
  // 9e5e8bc91e it is now aligned to lower_right which would be a BC break
  // for non-square masks. We will not support non-square masks for causal w/
  // FAV2
  if (params.is_causal && !params.query.is_nested() &&
      !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention XPU does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ",
          params.query.sym_size(-2),
          " seqlen_k: ",
          params.key.sym_size(-2),
          ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

inline bool check_flash_attention_deterministic(
    const sdp_params& params,
    bool debug) {
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (debug) {
      TORCH_WARN("Flash attention XPU is not deterministic.");
    }
    return false;
  }
  return true;
}

bool can_use_flash_attention(sdp_params const& params, bool debug) {
  constexpr auto constraints =
      std::array<bool (*)(sdp_params const&, bool), 14>{
          is_flash_attention_available,
          check_flash_attention_hardware_support,
          check_for_attn_mask,
          check_for_dropout,
          check_nested_tensor,
          check_tensor_shapes,
          check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
          check_nonzero_sequence_lengths_dense,
          check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
          check_flash_causal_non_square_seqlens,
          check_flash_attention_datatype,
          check_flash_attention_head_dim_size,
          check_flash_attention_layout,
          check_flash_attention_deterministic};
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return true;
}

} // namespace sdp
