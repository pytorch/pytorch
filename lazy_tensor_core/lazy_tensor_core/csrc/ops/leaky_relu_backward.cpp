#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LeakyReluBackward::LeakyReluBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& input, double negative_slope,
                                     bool self_is_result)
    : TsNode(torch::lazy::OpKind(at::aten::leaky_relu_backward), {grad_output, input},
           ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope),
      self_is_result_(self_is_result) {}

NodePtr LeakyReluBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<LeakyReluBackward>(operands.at(0), operands.at(1),
                                     negative_slope_, self_is_result_);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", negative_slope=" << negative_slope_
     << ", self_is_result=" << self_is_result_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
