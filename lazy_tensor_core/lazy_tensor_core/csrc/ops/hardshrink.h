#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Hardshrink : public TsNode {
 public:
  Hardshrink(const torch::lazy::Value& input, const at::Scalar& lambda);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  at::Scalar lambda() const { return lambda_; }

 private:
  at::Scalar lambda_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
