#include "lazy_tensor_core/csrc/ops/masked_fill.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaskedFill::MaskedFill(const torch::lazy::Value& input, const torch::lazy::Value& mask,
                       const at::Scalar& value)
    : TsNode(OpKind(at::aten::masked_fill), {input, mask}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

NodePtr MaskedFill::Clone(OpList operands) const {
  return torch::lazy::MakeNode<MaskedFill>(operands.at(0), operands.at(1), value_);
}

std::string MaskedFill::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
