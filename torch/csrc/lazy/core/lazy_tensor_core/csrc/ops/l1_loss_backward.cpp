#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

L1LossBackward::L1LossBackward(const Value& grad_output, const Value& input,
                               const Value& target, ReductionMode reduction)
    : Node(ir::OpKind(at::aten::l1_loss_backward), {grad_output, input, target},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(
               lazy_tensors::util::GetEnumValue(reduction))),
      reduction_(reduction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr L1LossBackward::Clone(OpList operands) const {
  return MakeNode<L1LossBackward>(operands.at(0), operands.at(1),
                                  operands.at(2), reduction_);
}

std::string L1LossBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
