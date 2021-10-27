#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AdaptiveAvgPool3d::AdaptiveAvgPool3d(const torch::lazy::Value& input,
                                     std::vector<int64_t> output_size)
    : TsNode(torch::lazy::OpKind(at::aten::adaptive_avg_pool3d), {input},
             /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AdaptiveAvgPool3d::Clone(OpList operands) const {
  return torch::lazy::MakeNode<AdaptiveAvgPool3d>(operands.at(0), output_size_);
}

std::string AdaptiveAvgPool3d::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
