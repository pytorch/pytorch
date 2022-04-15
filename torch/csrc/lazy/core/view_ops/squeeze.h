#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

TORCH_API std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim);

class TORCH_API Squeeze : public TsNode {
 public:
  // Squeeze out the specified dimension index, -1 for all trivial dimensions.
  Squeeze(const torch::lazy::Value& input, int dim);

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  int dim_;
};

}  // namespace lazy
}  // namespace torch
