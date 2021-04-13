#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NativeBatchNormBackward::NativeBatchNormBackward(
    const Value& grad_out, const Value& input, const Value& weight,
    const Value& save_mean, const Value& save_invstd, bool training, double eps)
    : Node(ir::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, save_mean, save_invstd},
           /*num_outputs=*/3, lazy_tensors::util::MHash(training, eps)),
      training_(training),
      eps_(eps) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NativeBatchNormBackward::Clone(OpList operands) const {
  return MakeNode<NativeBatchNormBackward>(operands.at(0), operands.at(1),
                                           operands.at(2), operands.at(3),
                                           operands.at(4), training_, eps_);
}

std::string NativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
