#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ThresholdBackward : public Node {
 public:
  ThresholdBackward(const Value& grad_output, const Value& input,
                    float threshold);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
