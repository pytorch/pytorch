#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LeakyReluBackward : public Node {
 public:
  LeakyReluBackward(const Value& grad_output, const Value& input,
                    double negative_slope);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  double negative_slope() const { return negative_slope_; }

 private:
  double negative_slope_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
