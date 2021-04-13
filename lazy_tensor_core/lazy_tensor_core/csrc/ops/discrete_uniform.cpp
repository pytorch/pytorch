#include "lazy_tensor_core/csrc/ops/discrete_uniform.h"

#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DiscreteUniform::DiscreteUniform(const Value& from, const Value& to,
                                 const Value& seed,
                                 const lazy_tensors::Shape& rng_shape)
    : Node(ir::OpKind(at::aten::random), {from, to, seed}, rng_shape,
           /*num_outputs=*/1, lazy_tensors::util::ShapeHash(rng_shape)) {}

NodePtr DiscreteUniform::Clone(OpList operands) const {
  return MakeNode<DiscreteUniform>(operands.at(0), operands.at(1),
                                   operands.at(2), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
