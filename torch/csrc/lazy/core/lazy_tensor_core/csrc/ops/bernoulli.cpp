#include "lazy_tensor_core/csrc/ops/bernoulli.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Bernoulli::Bernoulli(const Value& probability, const Value& seed,
                     lazy_tensors::Shape shape)
    : Node(ir::OpKind(at::aten::bernoulli), {probability, seed},
           std::move(shape)) {}

NodePtr Bernoulli::Clone(OpList operands) const {
  return MakeNode<Bernoulli>(operands.at(0), operands.at(1), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
