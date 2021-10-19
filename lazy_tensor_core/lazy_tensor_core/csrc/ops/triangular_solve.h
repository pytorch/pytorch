#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TriangularSolve : public TsNode {
 public:
  TriangularSolve(const torch::lazy::Value& rhs, const torch::lazy::Value& lhs, bool left_side,
                  bool lower, bool transpose, bool unit_diagonal);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool left_side() const { return left_side_; }

  bool lower() const { return lower_; }

  bool transpose() const { return transpose_; }

  bool unit_diagonal() const { return unit_diagonal_; }

 private:
  bool left_side_;
  bool lower_;
  bool transpose_;
  bool unit_diagonal_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
