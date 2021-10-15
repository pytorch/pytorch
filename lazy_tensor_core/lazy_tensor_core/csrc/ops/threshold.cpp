#include "lazy_tensor_core/csrc/ops/threshold.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Threshold::Threshold(const Value& input, float threshold, float value)
    : TsNode(ir::OpKind(at::aten::threshold), {input}, GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(threshold, value)),
      threshold_(threshold),
      value_(value) {}

NodePtr Threshold::Clone(OpList operands) const {
  return MakeNode<Threshold>(operands.at(0), threshold_, value_);
}

std::string Threshold::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", threshold=" << threshold_
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
