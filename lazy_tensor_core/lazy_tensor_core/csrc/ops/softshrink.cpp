#include "lazy_tensor_core/csrc/ops/softshrink.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Softshrink::Softshrink(const torch::lazy::Value& input, const at::Scalar& lambda)
    : TsNode(OpKind(at::aten::softshrink), {input}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Softshrink::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr Softshrink::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Softshrink>(operands.at(0), lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
