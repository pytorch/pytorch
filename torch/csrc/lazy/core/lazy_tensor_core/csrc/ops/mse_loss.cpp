#include "lazy_tensor_core/csrc/ops/mse_loss.h"

#include <ATen/core/Reduction.h>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MseLoss::MseLoss(const Value& input, const Value& target,
                 ReductionMode reduction)
    : Node(ir::OpKind(at::aten::mse_loss), {input, target},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(
               lazy_tensors::util::GetEnumValue(reduction))),
      reduction_(reduction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MseLoss::Clone(OpList operands) const {
  return MakeNode<MseLoss>(operands.at(0), operands.at(1), reduction_);
}

std::string MseLoss::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
