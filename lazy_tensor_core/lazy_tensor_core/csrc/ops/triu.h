#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Node for the upper triangular part of a matrix (2-D tensor) or batch of
// matrices input.
class Triu : public TsNode {
 public:
  Triu(const torch::lazy::Value& input, lazy_tensors::int64 diagonal);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 diagonal() const { return diagonal_; }

 private:
  lazy_tensors::int64 diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
