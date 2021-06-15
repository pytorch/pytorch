#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TSNativeBatchNormBackward::TSNativeBatchNormBackward(
    const Value& grad_out, const Value& input, const Value& weight,
    const Value& running_mean, const Value& running_var, const Value& save_mean,
    const Value& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask)
    : Node(ir::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, running_mean, running_var, save_mean,
            save_invstd},
           /*num_outputs=*/3,
           lazy_tensors::util::MHash(training, eps, output_mask[0],
                                     output_mask[1], output_mask[2])),
      training_(training),
      eps_(eps),
      output_mask_(output_mask) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

TSNativeBatchNormBackward::TSNativeBatchNormBackward(
    const Value& grad_out, const Value& input, const Value& weight,
    const Value& save_mean, const Value& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask)
    : Node(ir::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, save_mean, save_invstd},
           /*num_outputs=*/3,
           lazy_tensors::util::MHash(training, eps, output_mask[0],
                                     output_mask[1], output_mask[2])),
      training_(training),
      eps_(eps),
      output_mask_(output_mask) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr TSNativeBatchNormBackward::Clone(OpList operands) const {
  if (operands.size() == 5) {
    return MakeNode<TSNativeBatchNormBackward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), training_, eps_, output_mask_);
  }
  LTC_CHECK_EQ(operands.size(), 7);
  return MakeNode<TSNativeBatchNormBackward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), operands.at(6), training_, eps_,
      output_mask_);
}

std::string TSNativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
