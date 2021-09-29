#include "lazy_tensor_core/csrc/ops/threshold_backward.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ThresholdBackward::ThresholdBackward(const Value& grad_output,
                                     const Value& input, float threshold)
    : TsNode(ir::OpKind(at::aten::threshold_backward), {grad_output, input},
           input.shape(), /*num_outputs=*/1,
           torch::lazy::MHash(threshold)),
      threshold_(threshold) {}

NodePtr ThresholdBackward::Clone(OpList operands) const {
  return MakeNode<ThresholdBackward>(operands.at(0), operands.at(1),
                                     threshold_);
}

std::string ThresholdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
