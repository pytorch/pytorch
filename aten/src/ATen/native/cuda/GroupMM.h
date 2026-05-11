#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>
#include <vector>

namespace at::cuda::detail {
TORCH_API void bf16bf16_grouped_mm(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor& out);

TORCH_API void bf16bf16_foreach_mm(
    at::TensorList self_list,
    at::TensorList mat2_list,
    std::vector<at::Tensor>& outputs);
} // namespace at::cuda::detail
