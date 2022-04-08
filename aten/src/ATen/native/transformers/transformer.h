#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
Tensor transformer_encoder_layer_forward(
    const Tensor& src,
    const int64_t embed_dim,
    const int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const bool use_gelu,
    const bool norm_first,
    const double layer_norm_eps,
    const Tensor& layer_norm_weight_1,
    const Tensor& layer_norm_bias_1,
    const Tensor& layer_norm_weight_2,
    const Tensor& layer_norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const c10::optional<Tensor>& mask);
} // namespace native
} // namespace at
