#include "lazy_tensor_core/csrc/ops/replication_pad_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ReplicationPadBackward::ReplicationPadBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    std::vector<int64_t> padding)
    : TsNode(ltc_replication_pad_backward, {grad_output, input},
             /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ReplicationPadBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ReplicationPadBackward>(operands.at(0), operands.at(1),
                                          padding_);
}

std::string ReplicationPadBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", padding=(" << c10::Join(", ", padding_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
