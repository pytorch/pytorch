#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

TORCH_API std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

class TORCH_API Unsqueeze : public TsNode {
 public:
  Unsqueeze(const torch::lazy::Value& input, int dim);

  std::string ToString() const override;

  int dim() const {
    return dim_;
  }

 private:
  int dim_;
};

} // namespace lazy
} // namespace torch
