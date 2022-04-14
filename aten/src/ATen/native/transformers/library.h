#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace at {
namespace native {
std::tuple<Tensor, Tensor> multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask,
    bool need_weights=false,
    bool average_attn_weights=true);

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_op_cpu(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head);

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_op_cuda(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head);

Tensor ffn_cpu(
    const Tensor& input,
    const Tensor& w1,
    const Tensor& b1,
    const Tensor& w2,
    const Tensor& b2,
    bool use_gelu,
    bool add_norm);
} // namespace native
} // namespace at
