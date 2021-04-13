#include "lazy_tensor_core/csrc/ops/normal.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Normal::Normal(const Value& mean, const Value& std, const Value& seed)
    : Node(ir::OpKind(at::aten::normal), {mean, std, seed}, mean.shape()) {}

NodePtr Normal::Clone(OpList operands) const {
  return MakeNode<Normal>(operands.at(0), operands.at(1), operands.at(2));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
