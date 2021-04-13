#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class SymEig : public Node {
 public:
  SymEig(const Value& input, bool eigenvectors, bool lower);

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
