#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

HardtanhBackward::HardtanhBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                                   const at::Scalar& min_val,
                                   const at::Scalar& max_val)
    : TsNode(OpKind(at::aten::hardtanh_backward), {grad_output, input},
           ir::GetShapeFromTsValue(grad_output), /*num_outputs=*/1,
           torch::lazy::MHash(ScalarHash(min_val), ScalarHash(max_val))),
      min_val_(min_val),
      max_val_(max_val) {}

std::string HardtanhBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", min_val=" << min_val_
     << ", max_val=" << max_val_;
  return ss.str();
}

NodePtr HardtanhBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<HardtanhBackward>(operands.at(0), operands.at(1), min_val_,
                                    max_val_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
