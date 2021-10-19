#include "lazy_tensor_core/csrc/ops/cholesky.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Cholesky::Cholesky(const torch::lazy::Value& input, bool lower)
    : TsNode(torch::lazy::OpKind(at::aten::cholesky), {input}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(lower)),
      lower_(lower) {}

NodePtr Cholesky::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Cholesky>(operands.at(0), lower_);
}

std::string Cholesky::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", lower=" << lower_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
