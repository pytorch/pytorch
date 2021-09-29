#include "lazy_tensor_core/csrc/ops/scatter.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Scatter::Scatter(const Value& input, const Value& index, const Value& src,
                 lazy_tensors::int64 dim)
    : TsNode(ir::OpKind(at::aten::scatter), {input, index, src}, input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr Scatter::Clone(OpList operands) const {
  return MakeNode<Scatter>(operands.at(0), operands.at(1), operands.at(2),
                           dim_);
}

std::string Scatter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
