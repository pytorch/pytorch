#pragma once
#include <ATen/core/Tensor.h>
#include <c10/macros/Export.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

using fused_sdp_choice_fn = int64_t (*)(const Tensor& query_, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, c10::optional<double> scale);

DECLARE_DISPATCH(fused_sdp_choice_fn, _fused_sdp_choice_stub);

TORCH_API Tensor bmm_nt(const Tensor& a, const Tensor& b);
TORCH_API Tensor masked_softmax(
    Tensor& attn_scores,
    c10::optional<Tensor> attn_mask,
    const Tensor& query,
    c10::optional<int64_t> mask_type = {});

using transform_bias_rescale_qkv_fn = void(*)(
    at::ScalarType type,
    void* _q_k_v,
    const void* _qkv,
    const void* _qkv_bias,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t num_head);

DECLARE_DISPATCH(transform_bias_rescale_qkv_fn, transform_bias_rescale_qkv_stub);

TORCH_API Tensor transform0213_gemm_nt_bias(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& query);

TORCH_API Tensor bmm_nn(Tensor& out, const Tensor& a, const Tensor& b);

TORCH_API void debug_assert_shape(int line, const Tensor& t, c10::IntArrayRef shape);

TORCH_API Tensor qkv_projection(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const Tensor& qkv_weight);

using flash_attention_fn = void (*)(
    const Tensor& output, const Tensor& logsumexp,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k,
    int64_t& max_q, int64_t& max_k, const Tensor& philox_seed,
    const Tensor& philox_offset, const Tensor& debug_attn_mask,
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal, bool return_debug_mask,
    c10::optional<double> scale);

using flash_attention_backward_fn = void (*)(
    const Tensor& grad_q, const Tensor& grad_k,
    const Tensor& grad_v, const Tensor& grad_out,
    const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k,
    const int64_t max_q, const int64_t max_k,
    double dropout_p, bool is_causal,
    const Tensor& philox_seed, const Tensor& philox_offset,
    c10::optional<double> scale);

DECLARE_DISPATCH(flash_attention_fn, flash_attention_kernel);
DECLARE_DISPATCH(flash_attention_backward_fn, flash_attention_backward_kernel);

} // namespace native
} // namespace at
