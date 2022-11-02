#pragma once
#include <ATen/ATen.h>
#include <c10/macros/Export.h>

namespace at {
namespace native {

TORCH_API Tensor bmm_nt(const Tensor& a, const Tensor& b);
TORCH_API Tensor masked_softmax(
    Tensor& attn_scores,
    c10::optional<Tensor> attn_mask,
    const Tensor& query,
    c10::optional<int64_t> mask_type = NULL);

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

} // namespace native
} // namespace at
