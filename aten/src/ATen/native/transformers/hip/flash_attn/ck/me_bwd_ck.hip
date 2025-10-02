#include <ATen/native/transformers/hip/flash_attn/flash_common_hip.hpp>
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>

#if defined(USE_ROCM_CK_SDPA)
namespace pytorch_flash {
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
    at::Tensor philox_seed,
    at::Tensor philox_offset)
{

  const int non_null_window_left  = -1;
  const int non_null_window_right = -1;

  std::optional<at::Tensor> opt_dQ, opt_dK, opt_dV;
  opt_dQ = dq_;
  opt_dK = dk_;
  opt_dV = dv_;

  if(!cu_seqlens_q.has_value()) {
    auto
      [dQ,
       dK,
       dV,
       softmax_d,
       dBias] =
        mha_bwd_ck(
          dout,
          q,
          k,
          v,
          out,
          softmax_lse,
          opt_dQ,
          opt_dK,
          opt_dV,
          attn_bias,
          bias_requires_grad,
          grad_bias,
          p_dropout,
          scale,
          is_causal,
          non_null_window_left,
          non_null_window_right,
          deterministic,
          philox_seed,
          philox_offset);
    return std::make_tuple(std::move(dQ), std::move(dK), std::move(dV), std::move(dBias));

  } else {
    // cu_seqlens only has a value in the nested tensor path which CK does not support
    TORCH_CHECK(false, "Nested Tensors not supported with CK backend.");
    return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{}, at::Tensor{});
    // TODO: Fix nested tensor(varlen) path
    /*
    auto
      [dQ,
       dK,
       dV,
       softmax_d,
       dBias] =
        mha_varlen_bwd_ck(
          dout,
          q,
          k,
          v,
          out,
          softmax_lse,
          opt_dQ,
          opt_dK,
          opt_dV,
          cu_seqlens_q.value(),
          cu_seqlens_k.value(),
          attn_bias,
          bias_requires_grad,
          grad_bias,
          max_seqlen_q,
          max_seqlen_k,
          p_dropout,
          scale,
          zero_tensors,
          is_causal,
          non_null_window_left,
          non_null_window_right,
          deterministic,
          philox_seed,
          philox_offset);
    return std::make_tuple(std::move(dQ), std::move(dK), std::move(dV), std::move(dBias));
    */
  }
  return std::make_tuple(at::Tensor{}, at::Tensor{}, at::Tensor{}, at::Tensor{});
}

} // namespace pytorch_flash
#endif // USE_ROCM_CK_SDPA
