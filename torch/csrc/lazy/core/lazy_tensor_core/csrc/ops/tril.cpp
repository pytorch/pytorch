#include "lazy_tensor_core/csrc/ops/tril.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Tril::Tril(const Value& input, lazy_tensors::int64 diagonal)
    : Node(ir::OpKind(at::aten::tril), {input}, input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(diagonal)),
      diagonal_(diagonal) {}

NodePtr Tril::Clone(OpList operands) const {
  return MakeNode<Tril>(operands.at(0), diagonal_);
}

std::string Tril::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
