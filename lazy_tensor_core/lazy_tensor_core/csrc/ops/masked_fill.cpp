#include "lazy_tensor_core/csrc/ops/masked_fill.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaskedFill::MaskedFill(const Value& input, const Value& mask,
                       const at::Scalar& value)
    : Node(OpKind(at::aten::masked_fill), {input, mask}, input.shape(),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

NodePtr MaskedFill::Clone(OpList operands) const {
  return MakeNode<MaskedFill>(operands.at(0), operands.at(1), value_);
}

std::string MaskedFill::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
