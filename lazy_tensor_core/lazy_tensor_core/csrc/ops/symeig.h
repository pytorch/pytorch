#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class SymEig : public TsNode {
 public:
  SymEig(const torch::lazy::Value& input, bool eigenvectors, bool lower);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool eigenvectors() const { return eigenvectors_; }

  bool lower() const { return lower_; }

 private:
  bool eigenvectors_;
  bool lower_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
