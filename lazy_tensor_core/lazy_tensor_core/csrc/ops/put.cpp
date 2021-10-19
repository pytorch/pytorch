#include "lazy_tensor_core/csrc/ops/put.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Put::Put(const torch::lazy::Value& input, const torch::lazy::Value& index, const torch::lazy::Value& source,
         bool accumulate)
    : TsNode(torch::lazy::OpKind(at::aten::put), {input, index, source}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(accumulate)),
      accumulate_(accumulate) {}

NodePtr Put::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Put>(operands.at(0), operands.at(1), operands.at(2),
                       accumulate_);
}

std::string Put::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
