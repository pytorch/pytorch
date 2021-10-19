#include "lazy_tensor_core/csrc/ops/reflection_pad2d_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ReflectionPad2dBackward::ReflectionPad2dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    std::vector<lazy_tensors::int64> padding)
    : TsNode(OpKind(at::aten::reflection_pad2d_backward), {grad_output, input},
           /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ReflectionPad2dBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ReflectionPad2dBackward>(operands.at(0), operands.at(1),
                                           padding_);
}

std::string ReflectionPad2dBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", padding=("
     << lazy_tensors::StrJoin(padding_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
