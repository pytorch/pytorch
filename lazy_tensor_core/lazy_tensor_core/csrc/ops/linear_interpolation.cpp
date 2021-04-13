#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LinearInterpolation::LinearInterpolation(const Value& value,
                                         const Value& new_value, double alpha)
    : Node(ltc_moving_average, {value, new_value}, value.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(alpha)),
      alpha_(alpha) {}

NodePtr LinearInterpolation::Clone(OpList operands) const {
  return MakeNode<LinearInterpolation>(operands.at(0), operands.at(1), alpha_);
}

std::string LinearInterpolation::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", alpha=" << alpha_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
