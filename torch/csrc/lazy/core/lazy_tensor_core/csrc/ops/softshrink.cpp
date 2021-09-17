#include "lazy_tensor_core/csrc/ops/softshrink.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Softshrink::Softshrink(const Value& input, const at::Scalar& lambda)
    : Node(OpKind(at::aten::softshrink), {input}, input.shape(),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Softshrink::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr Softshrink::Clone(OpList operands) const {
  return MakeNode<Softshrink>(operands.at(0), lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
