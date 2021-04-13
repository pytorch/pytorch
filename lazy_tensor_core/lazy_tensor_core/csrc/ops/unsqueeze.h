#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Unsqueeze : public Node {
 public:
  // Insert a dimension of size one at the specified position.
  Unsqueeze(const Value& input, int dim);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  // Position to unsqueeze.
  int dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
