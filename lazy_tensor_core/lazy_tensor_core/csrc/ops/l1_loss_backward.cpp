#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

L1LossBackward::L1LossBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                               const torch::lazy::Value& target, ReductionMode reduction)
    : TsNode(torch::lazy::OpKind(at::aten::l1_loss_backward), {grad_output, input, target},
           /*num_outputs=*/1,
           torch::lazy::MHash(
               lazy_tensors::util::GetEnumValue(reduction))),
      reduction_(reduction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr L1LossBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<L1LossBackward>(operands.at(0), operands.at(1),
                                  operands.at(2), reduction_);
}

std::string L1LossBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
