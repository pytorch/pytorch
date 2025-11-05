#pragma once

#include <ATen/ATen.h>

namespace sycltla {

bool is_flash_attention_available();

std::tuple<at::Tensor, at::Tensor> flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const std::optional<at::Tensor>& attn_mask,
    const double dropout,
    const bool is_causal,
    const float scale);

} // namespace sycltla
