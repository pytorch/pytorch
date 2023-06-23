#pragma once
#include <cstddef>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace pytorch_fmha {

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        at::Tensor &out,
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q_,
        const int max_seqlen_k_,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool return_softmax,
        const int num_splits);

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
        const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &out,   // total_q x num_heads x head_size
        const at::Tensor &softmax_lse_,     // b x h x s softmax logsumexp
        at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q_,
        const int max_seqlen_k_,          // max sequence length to choose the kernel
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const int num_splits,
        at::Tensor philox_seed,
        at::Tensor philox_offset
);

} // namespace fmha
