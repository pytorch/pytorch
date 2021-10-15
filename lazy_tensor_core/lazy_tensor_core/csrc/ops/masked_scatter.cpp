#include "lazy_tensor_core/csrc/ops/masked_scatter.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaskedScatter::MaskedScatter(const Value& input, const Value& mask,
                             const Value& source)
    : TsNode(ir::OpKind(at::aten::masked_scatter), {input, mask, source},
           GetShapeFromTsValue(input),
           /*num_outputs=*/1) {}

NodePtr MaskedScatter::Clone(OpList operands) const {
  return MakeNode<MaskedScatter>(operands.at(0), operands.at(1),
                                 operands.at(2));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
