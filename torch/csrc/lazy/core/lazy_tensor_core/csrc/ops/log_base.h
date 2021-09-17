#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LogBase : public Node {
 public:
  LogBase(const Value& input, ir::OpKind kind, double base);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  double base() const { return base_; }

 private:
  double base_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
