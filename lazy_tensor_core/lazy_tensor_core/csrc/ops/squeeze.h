#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Squeeze : public TsNode {
 public:
  // Squeeze out the specified dimension index, -1 for all trivial dimensions.
  Squeeze(const torch::lazy::Value& input, int dim);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  int dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
