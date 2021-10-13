#include "lazy_tensor_core/csrc/ops/leaky_relu.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LeakyRelu::LeakyRelu(const Value& input, double negative_slope)
    : TsNode(ir::OpKind(at::aten::leaky_relu), {input}, GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

NodePtr LeakyRelu::Clone(OpList operands) const {
  return MakeNode<LeakyRelu>(operands.at(0), negative_slope_);
}

std::string LeakyRelu::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
