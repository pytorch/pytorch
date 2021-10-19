#include "lazy_tensor_core/csrc/ops/shrink_backward.h"

#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ShrinkBackward::ShrinkBackward(OpKind kind, const torch::lazy::Value& grad_output,
                               const torch::lazy::Value& input, const at::Scalar& lambda)
    : TsNode(kind, {grad_output, input}, ir::GetShapeFromTsValue(input), /*num_outputs=*/1,
           ScalarHash(lambda)),
      lambda_(std::move(lambda)) {}

std::string ShrinkBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", lambda=" << lambda_;
  return ss.str();
}

NodePtr ShrinkBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ShrinkBackward>(op(), operands.at(0), operands.at(1),
                                  lambda_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
