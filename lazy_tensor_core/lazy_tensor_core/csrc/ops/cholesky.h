#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Cholesky : public Node {
 public:
  Cholesky(const Value& input, bool lower);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool lower() const { return lower_; }

 private:
  bool lower_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
