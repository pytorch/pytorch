#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// IR node for the threshold operation.
class Threshold : public Node {
 public:
  Threshold(const Value& input, float threshold, float value);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

  float value() const { return value_; }

 private:
  float threshold_;
  float value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
