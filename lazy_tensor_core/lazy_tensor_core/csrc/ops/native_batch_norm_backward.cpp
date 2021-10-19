#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NativeBatchNormBackward::NativeBatchNormBackward(
    const torch::lazy::Value& grad_out, const torch::lazy::Value& input, const torch::lazy::Value& weight,
    const torch::lazy::Value& save_mean, const torch::lazy::Value& save_invstd, bool training, double eps)
    : TsNode(torch::lazy::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, save_mean, save_invstd},
           /*num_outputs=*/3, torch::lazy::MHash(training, eps)),
      training_(training),
      eps_(eps) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NativeBatchNormBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<NativeBatchNormBackward>(operands.at(0), operands.at(1),
                                           operands.at(2), operands.at(3),
                                           operands.at(4), training_, eps_);
}

std::string NativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
