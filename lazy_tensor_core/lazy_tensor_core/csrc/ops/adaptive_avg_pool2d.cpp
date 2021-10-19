#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AdaptiveAvgPool2d::AdaptiveAvgPool2d(
    const torch::lazy::Value& input, std::vector<lazy_tensors::int64> output_size)
    : TsNode(torch::lazy::OpKind(at::aten::adaptive_avg_pool2d), {input},
           /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AdaptiveAvgPool2d::Clone(OpList operands) const {
  return torch::lazy::MakeNode<AdaptiveAvgPool2d>(operands.at(0), output_size_);
}

std::string AdaptiveAvgPool2d::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=("
     << lazy_tensors::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
