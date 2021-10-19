#include "lazy_tensor_core/csrc/ops/hardshrink.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Hardshrink::Hardshrink(const torch::lazy::Value& input, const at::Scalar& lambda)
    : TsNode(OpKind(at::aten::hardshrink), {input}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string Hardshrink::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr Hardshrink::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Hardshrink>(operands.at(0), lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
