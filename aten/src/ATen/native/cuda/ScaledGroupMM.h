#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {

enum class GroupMMInputMatrixType {
  GroupMMInputMatrixType_MatrixA_2D_MatrixB_2D,
  GroupMMInputMatrixType_MatrixA_2D_MatrixB_3D,
  GroupMMInputMatrixType_MatrixA_3D_MatrixB_2D,
  GroupMMInputMatrixType_MatrixA_3D_MatrixB_3D,
};

struct GroupCountInfo {
  // This is the M,N,K of the real 2D tensor, not the stacked tensor
  int M;
  int N;
  int K;
  int group_count;
  GroupMMInputMatrixType input_matrix_type;
};

GroupCountInfo get_group_count(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs);

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
