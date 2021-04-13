#include "lazy_tensor_core/csrc/ops/shrink_backward.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ShrinkBackward::ShrinkBackward(OpKind kind, const Value& grad_output,
                               const Value& input, const at::Scalar& lambda)
    : Node(kind, {grad_output, input}, input.shape(), /*num_outputs=*/1,
           ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string ShrinkBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr ShrinkBackward::Clone(OpList operands) const {
  return MakeNode<ShrinkBackward>(op(), operands.at(0), operands.at(1),
                                  lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
