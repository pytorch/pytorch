#include "lazy_tensor_core/csrc/ops/masked_scatter.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaskedScatter::MaskedScatter(const torch::lazy::Value& input, const torch::lazy::Value& mask,
                             const torch::lazy::Value& source)
    : TsNode(torch::lazy::OpKind(at::aten::masked_scatter), {input, mask, source},
           ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1) {}

NodePtr MaskedScatter::Clone(OpList operands) const {
  return torch::lazy::MakeNode<MaskedScatter>(operands.at(0), operands.at(1),
                                 operands.at(2));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
