#include "lazy_tensor_core/csrc/ops/normal.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Normal::Normal(const torch::lazy::Value& mean, const torch::lazy::Value& std, const torch::lazy::Value& seed)
    : TsNode(torch::lazy::OpKind(at::aten::normal), {mean, std, seed}, ir::GetShapeFromTsValue(mean)) {}

NodePtr Normal::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Normal>(operands.at(0), operands.at(1), operands.at(2));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
