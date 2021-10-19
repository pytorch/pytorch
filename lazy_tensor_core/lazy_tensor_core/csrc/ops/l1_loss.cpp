#include "lazy_tensor_core/csrc/ops/l1_loss.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

L1Loss::L1Loss(const torch::lazy::Value& input, const torch::lazy::Value& target, ReductionMode reduction)
    : TsNode(torch::lazy::OpKind(at::aten::l1_loss), {input, target},
           /*num_outputs=*/1,
           torch::lazy::MHash(
               lazy_tensors::util::GetEnumValue(reduction))),
      reduction_(reduction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr L1Loss::Clone(OpList operands) const {
  return torch::lazy::MakeNode<L1Loss>(operands.at(0), operands.at(1), reduction_);
}

std::string L1Loss::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
