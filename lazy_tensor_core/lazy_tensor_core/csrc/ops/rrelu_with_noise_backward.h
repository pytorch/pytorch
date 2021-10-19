#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class RreluWithNoiseBackward : public TsNode {
 public:
  RreluWithNoiseBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                         const torch::lazy::Value& noise, const at::Scalar& lower,
                         const at::Scalar& upper, bool training);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const at::Scalar& lower() const { return lower_; }

  const at::Scalar& upper() const { return upper_; }

  bool training() const { return training_; }

 private:
  at::Scalar lower_;
  at::Scalar upper_;
  bool training_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
