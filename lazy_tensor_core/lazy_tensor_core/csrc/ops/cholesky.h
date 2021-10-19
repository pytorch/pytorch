#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Cholesky : public TsNode {
 public:
  Cholesky(const torch::lazy::Value& input, bool lower);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool lower() const { return lower_; }

 private:
  bool lower_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
