#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <ATen/Context.h>


namespace pytorch_flash {

// AOTriton Implementation
TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_aot(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
            const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
            std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
            std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
            const float p_dropout,
            const float softmax_scale,
            bool is_causal,
            int window_size_left,
            int window_size_right,
            const bool return_softmax,
            std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_aot(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                   const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                   const at::Tensor &cu_seqlens_q,  // b+1
                   const at::Tensor &cu_seqlens_k,  // b+1
                   std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                   std::optional<at::Tensor> &block_table_,
                   std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                   int max_seqlen_q,
                   const int max_seqlen_k,
                   const float p_dropout,
                   const float softmax_scale,
                   const bool zero_tensors,
                   bool is_causal,
                   int window_size_left,
                   int window_size_right,
                   const bool return_softmax,
                   std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd_aot(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
            const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
            const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
            const at::Tensor &softmax_lse,     // b x h x seqlen_q
            std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
            std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
            std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
            std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
            const float p_dropout,         // probability to drop
            const float softmax_scale,
            const bool is_causal,
            int window_size_left,
            int window_size_right,
            const bool deterministic,
            const at::Tensor philox_seed,
            const at::Tensor philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd_aot(const at::Tensor &dout,  // total_q x num_heads, x head_size
                   const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                   const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   const at::Tensor &out,   // total_q x num_heads x head_size
                   const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
                   std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                   std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                   const at::Tensor &cu_seqlens_q,  // b+1
                   const at::Tensor &cu_seqlens_k,  // b+1
                   std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                   const int max_seqlen_q,
                   const int max_seqlen_k,          // max sequence length to choose the kernel
                   const float p_dropout,         // probability to drop
                   const float softmax_scale,
                   const bool zero_tensors,
                   const bool is_causal,
                   int window_size_left,
                   int window_size_right,
                   const bool deterministic,
                   const at::Tensor philox_seed,
                   const at::Tensor philox_offset);
#if defined(USE_CK_FLASH_ATTENTION)
// CK implementation
TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_ck(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
           const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
           const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
           std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
           std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
           const float p_dropout,
           const float softmax_scale,
           bool is_causal,
           int window_size_left,
           int window_size_right,
           const bool return_softmax,
           std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd_ck(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                  const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                  const at::Tensor &cu_seqlens_q,  // b+1
                  const at::Tensor &cu_seqlens_k,  // b+1
                  std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                  std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                  int max_seqlen_q,
                  const int max_seqlen_k,
                  const float p_dropout,
                  const float softmax_scale,
                  const bool zero_tensors,
                  bool is_causal,
                  int window_size_left,
                  int window_size_right,
                  const bool return_softmax,
                  std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd_ck(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
           const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
           const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
           const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
           const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
           const at::Tensor &softmax_lse,     // b x h x seqlen_q
           std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
           std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
           std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
           std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
           const float p_dropout,         // probability to drop
           const float softmax_scale,
           const bool is_causal,
           int window_size_left,
           int window_size_right,
           const bool deterministic,
           const at::Tensor philox_seed,
           const at::Tensor philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd_ck(const at::Tensor &dout,  // total_q x num_heads, x head_size
                  const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                  const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  const at::Tensor &out,   // total_q x num_heads x head_size
                  const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
                  std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                  std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                  const at::Tensor &cu_seqlens_q,  // b+1
                  const at::Tensor &cu_seqlens_k,  // b+1
                  std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                  const int max_seqlen_q,
                  const int max_seqlen_k,          // max sequence length to choose the kernel
                  const float p_dropout,         // probability to drop
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
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
#if defined(USE_CK_FLASH_ATTENTION)
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    return mha_fwd_ck(q,
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
  } else {
    return mha_fwd_aot(q,
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
#else
     return mha_fwd_aot(q,
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
#endif
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_, // Not used on ROCm. Keeping for parity with CUDA
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool return_softmax,
               std::optional<at::Generator> gen_) {
#if defined(USE_CK_FLASH_ATTENTION)
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    return mha_varlen_fwd_ck(q,
                             k,
                             v,
                             out_,
                             cu_seqlens_q,
                             cu_seqlens_k,
                             seqused_k,
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
  } else {
    return mha_varlen_fwd_aot(q,
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
#else
    return mha_varlen_fwd_aot(q,
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
#endif

}


inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {
#if defined(USE_CK_FLASH_ATTENTION)
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    return mha_bwd_ck(dout,
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
  } else {
    return mha_bwd_aot(dout,
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
#else
    return mha_bwd_aot(dout,
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
#endif

}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset) {
#if defined(USE_CK_FLASH_ATTENTION)
  if(at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Ck) {
    return mha_varlen_bwd_ck(dout,
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
  } else {
    return mha_varlen_bwd_aot(dout,
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
#else
    return mha_varlen_bwd_aot(dout,
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
#endif
}

} // namespace pytorch_flash
