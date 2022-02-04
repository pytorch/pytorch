#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

TORCH_API std::tuple<at::Tensor, at::Tensor> buildL2NormMpCuda();

class TORCH_API L2NormMpCuda : public torch::lazy::TsNode {
 public:
  L2NormMpCuda(OpList tensor_args, const std::vector<int64_t>& list_sizes, int chunk_size, c10::optional<bool> per_tensor_python);

  torch::lazy::TSOpVector Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const override;

  std::string ToString() const override;
  std::vector<int64_t> list_sizes_;
  int chunk_size_;
  c10::optional<bool> per_tensor_python_;
};

}  // namespace ops
}  // namespace ir