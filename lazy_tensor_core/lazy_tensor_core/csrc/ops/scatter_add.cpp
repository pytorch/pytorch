#include "lazy_tensor_core/csrc/ops/scatter_add.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ScatterAdd::ScatterAdd(const Value& input, const Value& index, const Value& src,
                       lazy_tensors::int64 dim)
    : TsNode(ir::OpKind(at::aten::scatter_add), {input, index, src},
           input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr ScatterAdd::Clone(OpList operands) const {
  return MakeNode<ScatterAdd>(operands.at(0), operands.at(1), operands.at(2),
                              dim_);
}

std::string ScatterAdd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
