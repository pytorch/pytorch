#pragma once
#include <cstddef>

#include <ATen/core/Tensor.h>

#if defined(USE_CK_FLASH_ATTENTION)
namespace pytorch_flash {

std::tuple<
    at::Tensor, // output
    at::Tensor, // q
    at::Tensor, // k
    at::Tensor, // v
    at::Tensor, // lse
    at::Tensor, // seed
    at::Tensor, // offset
    at::Tensor> // dropout randval
mem_eff_forward_ck(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    float p_dropout,
    bool return_dropout_randval,
    std::optional<bool> is_causal,
    std::optional<float> scale,
    const std::optional<at::Tensor>& attn_bias_,
    std::optional<at::Tensor>& out_,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    const std::optional<at::Tensor>& seqstart_q,
    const std::optional<at::Tensor>& seqstart_k,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& seqused_k_
);

std::tuple<
    at::Tensor, // dQ
    at::Tensor, // dK
    at::Tensor, // dV
    at::Tensor> // dBias
mem_eff_backward_ck(
    const at::Tensor &dout,
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &softmax_lse,
    const at::Tensor &dq_,
    const at::Tensor &dk_,
    const at::Tensor &dv_,
    std::optional<at::Tensor> &attn_bias,
    bool bias_requires_grad,
    std::optional<at::Tensor> &grad_bias,
    std::optional<at::Tensor> &cu_seqlens_q,
    std::optional<at::Tensor> &cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float scale,
    bool is_causal,
    bool deterministic,
    bool zero_tensors,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset);

} // namespace pytorch_flash
#endif // USE_CK_FLASH_ATTENTION
