#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {
TORCH_API void f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out);
} // namespace at::cuda::detail
