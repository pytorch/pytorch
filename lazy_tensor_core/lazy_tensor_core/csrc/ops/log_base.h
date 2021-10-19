#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LogBase : public TsNode {
 public:
  LogBase(const torch::lazy::Value& input, torch::lazy::OpKind kind, double base);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  double base() const { return base_; }

 private:
  double base_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
