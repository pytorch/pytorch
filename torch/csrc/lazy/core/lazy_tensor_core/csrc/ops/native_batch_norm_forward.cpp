#include "lazy_tensor_core/csrc/ops/native_batch_norm_forward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NativeBatchNormForward::NativeBatchNormForward(const Value& input,
                                               const Value& weight,
                                               const Value& bias,
                                               const Value& running_mean,
                                               const Value& running_var,
                                               bool training, double eps)
    : Node(ir::OpKind(at::aten::native_batch_norm),
           {input, weight, bias, running_mean, running_var},
           /*num_outputs=*/4, lazy_tensors::util::MHash(training, eps)),
      training_(training),
      eps_(eps) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NativeBatchNormForward::Clone(OpList operands) const {
  return MakeNode<NativeBatchNormForward>(operands.at(0), operands.at(1),
                                          operands.at(2), operands.at(3),
                                          operands.at(4), training_, eps_);
}

std::string NativeBatchNormForward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
