#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class QR : public Node {
 public:
  QR(const Value& input, bool some);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool some() const { return some_; }

 private:
  bool some_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
