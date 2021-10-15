#include "lazy_tensor_core/csrc/ops/exponential.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Exponential::Exponential(const Value& lambda, const Value& seed,
                         lazy_tensors::Shape shape)
    : TsNode(ir::OpKind(at::aten::exponential), {lambda, seed},
           std::move(shape)) {}

NodePtr Exponential::Clone(OpList operands) const {
  return MakeNode<Exponential>(operands.at(0), operands.at(1), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
