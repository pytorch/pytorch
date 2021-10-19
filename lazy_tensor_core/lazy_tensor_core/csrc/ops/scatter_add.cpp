#include "lazy_tensor_core/csrc/ops/scatter_add.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ScatterAdd::ScatterAdd(const torch::lazy::Value& input, const torch::lazy::Value& index, const torch::lazy::Value& src,
                       lazy_tensors::int64 dim)
    : TsNode(torch::lazy::OpKind(at::aten::scatter_add), {input, index, src},
           ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr ScatterAdd::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ScatterAdd>(operands.at(0), operands.at(1), operands.at(2),
                              dim_);
}

std::string ScatterAdd::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
