#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MaskedFill : public TsNode {
 public:
  MaskedFill(const torch::lazy::Value& input, const torch::lazy::Value& mask, const at::Scalar& value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  at::Scalar value() const { return value_; }

 private:
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
