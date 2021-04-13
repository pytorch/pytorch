#include "lazy_tensor_core/csrc/ops/threshold.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Threshold::Threshold(const Value& input, float threshold, float value)
    : Node(ir::OpKind(at::aten::threshold), {input}, input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(threshold, value)),
      threshold_(threshold),
      value_(value) {}

NodePtr Threshold::Clone(OpList operands) const {
  return MakeNode<Threshold>(operands.at(0), threshold_, value_);
}

std::string Threshold::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
