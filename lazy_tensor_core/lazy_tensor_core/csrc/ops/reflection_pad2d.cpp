#include "lazy_tensor_core/csrc/ops/reflection_pad2d.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ReflectionPad2d::ReflectionPad2d(const torch::lazy::Value& input,
                                 std::vector<int64_t> padding)
    : TsNode(OpKind(at::aten::reflection_pad2d), {input},
             /*num_outputs=*/1, torch::lazy::MHash(padding)),
      padding_(std::move(padding)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ReflectionPad2d::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ReflectionPad2d>(operands.at(0), padding_);
}

std::string ReflectionPad2d::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", padding=(" << c10::Join(", ", padding_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
