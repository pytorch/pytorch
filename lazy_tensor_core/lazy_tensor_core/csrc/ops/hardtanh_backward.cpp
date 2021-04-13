#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

HardtanhBackward::HardtanhBackward(const Value& grad_output, const Value& input,
                                   const at::Scalar& min_val,
                                   const at::Scalar& max_val)
    : Node(OpKind(at::aten::hardtanh_backward), {grad_output, input},
           grad_output.shape(), /*num_outputs=*/1,
           lazy_tensors::util::MHash(ScalarHash(min_val), ScalarHash(max_val))),
      min_val_(min_val),
      max_val_(max_val) {}

std::string HardtanhBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", min_val=" << min_val_
     << ", max_val=" << max_val_;
  return ss.str();
}

NodePtr HardtanhBackward::Clone(OpList operands) const {
  return MakeNode<HardtanhBackward>(operands.at(0), operands.at(1), min_val_,
                                    max_val_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
