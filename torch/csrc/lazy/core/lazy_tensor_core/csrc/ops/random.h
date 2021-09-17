#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Random : public Node {
 public:
  Random(const Value& input);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
