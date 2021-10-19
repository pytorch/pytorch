#include "lazy_tensor_core/csrc/ops/flip.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Flip::Flip(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> dims)
    : TsNode(torch::lazy::OpKind(at::aten::flip), {input}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {}

NodePtr Flip::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Flip>(operands.at(0), dims_);
}

std::string Flip::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dims=(" << lazy_tensors::StrJoin(dims_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
