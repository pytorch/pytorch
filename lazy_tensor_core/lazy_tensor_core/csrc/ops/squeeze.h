#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

std::vector<int64_t> BuildSqueezedDimensions(c10::ArrayRef<int64_t> dimensions,
                                             int64_t squeeze_dim);

class Squeeze : public TsNode {
 public:
  // Squeeze out the specified dimension index, -1 for all trivial dimensions.
  Squeeze(const torch::lazy::Value& input, int dim);

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  int dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
