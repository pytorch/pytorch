#include "lazy_tensor_core/csrc/ops/triu.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Triu::Triu(const Value& input, lazy_tensors::int64 diagonal)
    : TsNode(ir::OpKind(at::aten::triu), {input}, GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Triu::Clone(OpList operands) const {
  return MakeNode<Triu>(operands.at(0), diagonal_);
}

std::string Triu::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
