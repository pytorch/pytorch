#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Normal : public Node {
 public:
  Normal(const Value& mean, const Value& std, const Value& seed);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
