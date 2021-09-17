#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LinearInterpolation : public Node {
 public:
  LinearInterpolation(const Value& value, const Value& new_value, double alpha);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  double alpha() const { return alpha_; }

 private:
  double alpha_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
