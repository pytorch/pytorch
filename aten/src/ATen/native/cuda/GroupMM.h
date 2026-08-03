#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {
TORCH_API void bf16bf16_grouped_mm(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out);
} // namespace at::cuda::detail
