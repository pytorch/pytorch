#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {
TORCH_API void f8f8bf16_grouped_mm(
    at::Tensor mat_a, // FP8
    at::Tensor mat_b, // FP8
    at::Tensor scale_a, // FP32
    at::Tensor scale_b, // FP32
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out);
} // namespace at::cuda::detail
