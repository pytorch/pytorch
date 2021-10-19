#include "lazy_tensor_core/csrc/ops/scatter.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Scatter::Scatter(const torch::lazy::Value& input, const torch::lazy::Value& index, const torch::lazy::Value& src,
                 lazy_tensors::int64 dim)
    : TsNode(torch::lazy::OpKind(at::aten::scatter), {input, index, src}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr Scatter::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Scatter>(operands.at(0), operands.at(1), operands.at(2),
                           dim_);
}

std::string Scatter::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
