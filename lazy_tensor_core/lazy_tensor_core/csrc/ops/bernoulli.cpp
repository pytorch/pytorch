#include "lazy_tensor_core/csrc/ops/bernoulli.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Bernoulli::Bernoulli(const torch::lazy::Value& probability, const torch::lazy::Value& seed,
                     lazy_tensors::Shape shape)
    : TsNode(torch::lazy::OpKind(at::aten::bernoulli), {probability, seed},
           std::move(shape)) {}

NodePtr Bernoulli::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Bernoulli>(operands.at(0), operands.at(1), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
