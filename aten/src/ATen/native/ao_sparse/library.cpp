#include <torch/library.h>

#include <torch/custom_class.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>

namespace ao {
namespace sparse {
torch::class_<LinearPackedParamsBase> register_linear_params();
}}

// Register operators
TORCH_LIBRARY(sparse, m) {
  ao::sparse::register_linear_params();

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_relu_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse::qlinear_unpack(__torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)"));
}
