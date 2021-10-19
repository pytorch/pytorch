#include "lazy_tensor_core/csrc/ops/threshold_backward.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ThresholdBackward::ThresholdBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& input, float threshold)
    : TsNode(torch::lazy::OpKind(at::aten::threshold_backward), {grad_output, input},
           ir::GetShapeFromTsValue(input), /*num_outputs=*/1,
           torch::lazy::MHash(threshold)),
      threshold_(threshold) {}

NodePtr ThresholdBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ThresholdBackward>(operands.at(0), operands.at(1),
                                     threshold_);
}

std::string ThresholdBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", threshold=" << threshold_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
