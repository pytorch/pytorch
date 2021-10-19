#include "lazy_tensor_core/csrc/ops/exponential.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Exponential::Exponential(const torch::lazy::Value& lambda, const torch::lazy::Value& seed,
                         lazy_tensors::Shape shape)
    : TsNode(torch::lazy::OpKind(at::aten::exponential), {lambda, seed},
           std::move(shape)) {}

NodePtr Exponential::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Exponential>(operands.at(0), operands.at(1), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
