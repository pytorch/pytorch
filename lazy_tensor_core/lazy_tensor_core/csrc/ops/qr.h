#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class QR : public TsNode {
 public:
  QR(const torch::lazy::Value& input, bool some);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool some() const { return some_; }

 private:
  bool some_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
