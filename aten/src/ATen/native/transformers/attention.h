#pragma once

#include <c10/util/Optional.h>

namespace at {
class Tensor;
namespace native {
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_cuda(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head);
void masked_softmax_dropout_cuda(
    const Tensor& attn_scores,
    const c10::optional<Tensor>& attn_mask);
} // namespace native
} // namespace at
