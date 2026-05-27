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

TORCH_API std::vector<at::Tensor> grouped_mm_from_ptrs(
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    int64_t M, int64_t N, int64_t K, int64_t G,
    int64_t lda, int64_t ldb);
} // namespace at::cuda::detail
