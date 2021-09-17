#include "lazy_tensor_core/csrc/ops/hardshrink.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Hardshrink::Hardshrink(const Value& input, const at::Scalar& lambda)
    : Node(OpKind(at::aten::hardshrink), {input}, input.shape(),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Hardshrink::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr Hardshrink::Clone(OpList operands) const {
  return MakeNode<Hardshrink>(operands.at(0), lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
