#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LinearInterpolation::LinearInterpolation(const torch::lazy::Value& value,
                                         const torch::lazy::Value& new_value, double alpha)
    : TsNode(ltc_moving_average, {value, new_value}, ir::GetShapeFromTsValue(value),
           /*num_outputs=*/1, torch::lazy::MHash(alpha)),
      alpha_(alpha) {}

NodePtr LinearInterpolation::Clone(OpList operands) const {
  return torch::lazy::MakeNode<LinearInterpolation>(operands.at(0), operands.at(1), alpha_);
}

std::string LinearInterpolation::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", alpha=" << alpha_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
