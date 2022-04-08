#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace at {
namespace native {
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_op_cuda(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head);
} // namespace native
} // namespace at
