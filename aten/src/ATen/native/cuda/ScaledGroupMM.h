#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {

enum class GroupMMInputMatrixType {
  MatrixA_2D_MatrixB_2D,
  MatrixA_2D_MatrixB_3D,
  MatrixA_3D_MatrixB_2D,
  MatrixA_3D_MatrixB_3D,
};

// Dimensions of individual 2D matrices within the grouped operation
struct GroupCountInfo {
  int M;
  int N;
  int K;
  int group_count;
  GroupMMInputMatrixType input_matrix_type;
};

// Extract group information from input tensors (maybe 2D or 3D) for grouped matrix multiplication
//
// For 3D tensors: group_count is derived from the first dimension (size(0))
// For 2D tensors: group_count is determined from the offsets tensor length
//
// Returns M, N, K dimensions representing the individual 2D matrix sizes within each group,
// not the dimensions of the stacked input tensors
//
// Notice: If the input tensors are 2D, the M, N, K may need to be recalculated from the offs tensor
GroupCountInfo get_group_info(
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
