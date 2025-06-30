#pragma once
#include <cstddef>

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  TORCH_CHECK(                         \
      uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")

#define ASSIGN_CHECK_OVERFLOW(A, B)                                    \
  {                                                                    \
    A = B;                                                             \
    TORCH_CHECK(                                                    \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

namespace pytorch_flash {

// AOTriton Implementation
TORCH_API
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_fwd_aot(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool return_softmax,
    const std::optional<at::Generator>& gen_);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_varlen_fwd_aot(
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>&
        seqused_k, // b. If given, only this many elements of each batch
                   // element's keys are used.
    std::optional<at::Tensor>& block_table_,
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool return_softmax,
    const std::optional<at::Generator>& gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd_aot(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& out, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>&
        dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool deterministic,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_varlen_bwd_aot(
    const at::Tensor& dout, // total_q x num_heads, x head_size
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& out, // total_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x s   softmax logsumexp
    std::optional<at::Tensor>&
        dq_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dk_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dv_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or b x num_heads
    const int max_seqlen_q,
    const int max_seqlen_k, // max sequence length to choose the kernel
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const bool deterministic,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset);

#if defined(USE_CK_FLASH_ATTENTION)
// CK implementation
TORCH_API
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_fwd_ck(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        out_, // batch_size x seqlen_q x num_heads x head_size
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool return_softmax,
    std::optional<at::Generator> gen_,
    const std::optional<at::Tensor>& attn_bias_); // batch_size x nheads x seqlen_q x seqlen_k

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_varlen_fwd_ck(
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>&
        seqused_k, // b. If given, only this many elements of each batch
                   // element's keys are used.
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool return_softmax,
    std::optional<at::Generator> gen_,
    const std::optional<at::Tensor>& attn_bias_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd_ck(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& out, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>&
        dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        attn_bias_, // batch_size x num_heads x seqlen_q x seqlen_k
    bool bias_requires_grad,
    std::optional<at::Tensor>& grad_bias,
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool deterministic,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_varlen_bwd_ck(
    const at::Tensor& dout, // total_q x num_heads, x head_size
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& out, // total_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x s   softmax logsumexp
    std::optional<at::Tensor>&
        dq_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dk_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dv_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>& attn_bias_, // num_heads or b x num_heads
    bool bias_requires_grad,
    std::optional<at::Tensor>& grad_bias,
    const int max_seqlen_q,
    const int max_seqlen_k, // max sequence length to choose the kernel
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool deterministic,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset);
#endif

TORCH_API
inline std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_fwd(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {
#if defined(USE_CK_FLASH_ATTENTION)
  if (at::globalContext().getROCmFAPreferredBackend() ==
      at::ROCmFABackend::Ck) {
    const int non_null_window_left = window_size_left.value_or(-1);
    const int non_null_window_right = window_size_right.value_or(-1);
    std::optional<at::Tensor> dummy_attn_bias = std::nullopt;
    return mha_fwd_ck(
        q,
        k,
        v,
        out_,
        p_dropout,
        softmax_scale,
        is_causal,
        non_null_window_left,
        non_null_window_right,
        return_softmax,
        gen_,
        dummy_attn_bias); // Not used in flash attention
  }
#endif
  return mha_fwd_aot(
      q,
      k,
      v,
      out_,
      alibi_slopes_,
      p_dropout,
      softmax_scale,
      is_causal,
      window_size_left,
      window_size_right,
      return_softmax,
      gen_);
}

inline std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_varlen_fwd(
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>&
        seqused_k, // b. If given, only this many elements of each batch
                   // element's keys are used.
    std::optional<at::Tensor>&
        block_table_, // Not used on ROCm. Keeping for parity with CUDA
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {
#if defined(USE_CK_FLASH_ATTENTION)
  if (at::globalContext().getROCmFAPreferredBackend() ==
      at::ROCmFABackend::Ck) {
    std::optional<at::Tensor> dummy_attn_bias = std::nullopt;
    const int non_null_window_left = window_size_left.value_or(-1);
    const int non_null_window_right = window_size_right.value_or(-1);
    return mha_varlen_fwd_ck(
        q,
        k,
        v,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        non_null_window_left,
        non_null_window_right,
        return_softmax,
        gen_,
        dummy_attn_bias); // Not used in flash attention
  }
#endif
  return mha_varlen_fwd_aot(
      q,
      k,
      v,
      out_,
      cu_seqlens_q,
      cu_seqlens_k,
      seqused_k,
      block_table_,
      alibi_slopes_,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale,
      zero_tensors,
      is_causal,
      window_size_left,
      window_size_right,
      return_softmax,
      gen_);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& out, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>&
        dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const float softcap,
    const bool deterministic,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset) {
  if (at::globalContext().getROCmFAPreferredBackend() ==
      at::ROCmFABackend::Ck) {
#if defined(USE_CK_FLASH_ATTENTION)
    std::optional<at::Tensor> non_null_dbias = std::nullopt;
    const int non_null_window_left = window_size_left.value_or(-1);
    const int non_null_window_right = window_size_right.value_or(-1);
    auto[dQuery,
         dKey,
         dValue,
         dSoftmax,
         dBias] = mha_bwd_ck(
                             dout,
                             q,
                             k,
                             v,
                             out,
                             softmax_lse,
                             dq_,
                             dk_,
                             dv_,
                             alibi_slopes_,
                             false,              // bias_requires_grad
                             non_null_dbias,
                             p_dropout,
                             softmax_scale,
                             is_causal,
                             non_null_window_left,
                             non_null_window_right,
                             deterministic,
                             philox_seed,
                             philox_offset);
    // for FA return [dQ, dV, dK, dSoftmax]
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue), std::move(dSoftmax));
#else
    TORCH_WARN_ONCE("Warning! You have opted to use CK flash attention backend in a build that was not compiled using USE_CK_FLASH_ATTENTION=1. Please set this variable and try again. Defaulting to use aotriton backend...");
#endif
  }
  return mha_bwd_aot(
      dout,
      q,
      k,
      v,
      out,
      softmax_lse,
      dq_,
      dk_,
      dv_,
      alibi_slopes_,
      p_dropout,
      softmax_scale,
      is_causal,
      window_size_left,
      window_size_right,
      deterministic,
      philox_seed,
      philox_offset);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_varlen_bwd(
    const at::Tensor& dout, // total_q x num_heads, x head_size
    const at::Tensor&
        q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor&
        k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor&
        v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& out, // total_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x s   softmax logsumexp
    std::optional<at::Tensor>&
        dq_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dk_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor>&
        dv_, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q, // b+1
    const at::Tensor& cu_seqlens_k, // b+1
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or b x num_heads
    const int max_seqlen_q,
    const int max_seqlen_k, // max sequence length to choose the kernel
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const float softcap,
    const bool deterministic,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset) {
#if defined(USE_CK_FLASH_ATTENTION)
  if (at::globalContext().getROCmFAPreferredBackend() ==
      at::ROCmFABackend::Ck) {
    std::optional<at::Tensor> non_null_dbias = std::nullopt;
    const int non_null_window_left = window_size_left.value_or(-1);
    const int non_null_window_right = window_size_right.value_or(-1);
    auto[dQuery,
         dKey,
         dValue,
         dSoftmax,
         dBias] = mha_varlen_bwd_ck(
                                    dout,
                                    q,
                                    k,
                                    v,
                                    out,
                                    softmax_lse,
                                    dq_,
                                    dk_,
                                    dv_,
                                    cu_seqlens_q,
                                    cu_seqlens_k,
                                    alibi_slopes_,
                                    false,          // bias_requires_grad
                                    non_null_dbias,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    p_dropout,
                                    softmax_scale,
                                    zero_tensors,
                                    is_causal,
                                    non_null_window_left,
                                    non_null_window_right,
                                    deterministic,
                                    philox_seed,
                                    philox_offset);
    // for FA return [dQ, dV, dK, dSoftmax]
    return std::make_tuple(std::move(dQuery), std::move(dKey), std::move(dValue), std::move(dSoftmax));
  }
#endif
  return mha_varlen_bwd_aot(
      dout,
      q,
      k,
      v,
      out,
      softmax_lse,
      dq_,
      dk_,
      dv_,
      cu_seqlens_q,
      cu_seqlens_k,
      alibi_slopes_,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale,
      zero_tensors,
      is_causal,
      window_size_left,
      window_size_right,
      deterministic,
      philox_seed,
      philox_offset);
}

} // namespace pytorch_flash
