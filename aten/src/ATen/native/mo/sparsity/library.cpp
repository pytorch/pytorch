#include <torch/library.h>

#include <torch/custom_class.h>
#include <ATen/native/mo/sparsity/cpu/packed_params.h>

torch::class_<SparseLinearPackedParamsBase> register_sparse_linear_params();

// Register operators
TORCH_LIBRARY(sparsity, m) {
  register_sparse_linear_params();

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear(Tensor X, __torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear_relu(Tensor X, __torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear_dynamic(Tensor X, __torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear_relu_dynamic(Tensor X, __torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack) -> Tensor Y"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparsity::sparse_qlinear_unpack(__torch__.torch.classes.sparsity.SparseLinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)"));
}
