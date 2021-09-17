#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LeakyReluBackward::LeakyReluBackward(const Value& grad_output,
                                     const Value& input, double negative_slope,
                                     bool self_is_result)
    : Node(ir::OpKind(at::aten::leaky_relu_backward), {grad_output, input},
           input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(negative_slope)),
      negative_slope_(negative_slope),
      self_is_result_(self_is_result) {}

NodePtr LeakyReluBackward::Clone(OpList operands) const {
  return MakeNode<LeakyReluBackward>(operands.at(0), operands.at(1),
                                     negative_slope_, self_is_result_);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", negative_slope=" << negative_slope_
     << ", self_is_result=" << self_is_result_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
