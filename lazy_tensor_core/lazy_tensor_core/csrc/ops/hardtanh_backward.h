#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class HardtanhBackward : public TsNode {
 public:
  HardtanhBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                   const at::Scalar& min_val, const at::Scalar& max_val);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  at::Scalar min_val() const { return min_val_; }

  at::Scalar max_val() const { return max_val_; }

 private:
  at::Scalar min_val_;
  at::Scalar max_val_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
