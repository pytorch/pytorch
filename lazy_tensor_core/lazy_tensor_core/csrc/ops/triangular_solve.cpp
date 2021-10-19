#include "lazy_tensor_core/csrc/ops/triangular_solve.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TriangularSolve::TriangularSolve(const torch::lazy::Value& rhs, const torch::lazy::Value& lhs,
                                 bool left_side, bool lower, bool transpose,
                                 bool unit_diagonal)
    : TsNode(torch::lazy::OpKind(at::aten::triangular_solve), {rhs, lhs},
           /*num_outputs=*/2,
           torch::lazy::MHash(left_side, lower, transpose,
                                     unit_diagonal)),
      left_side_(left_side),
      lower_(lower),
      transpose_(transpose),
      unit_diagonal_(unit_diagonal) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr TriangularSolve::Clone(OpList operands) const {
  return torch::lazy::MakeNode<TriangularSolve>(operands.at(0), operands.at(1), left_side_,
                                   lower_, transpose_, unit_diagonal_);
}

std::string TriangularSolve::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", left_side=" << left_side_ << ", lower=" << lower_
     << ", transpose=" << transpose_ << ", unit_diagonal=" << unit_diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
