#include "lazy_tensor_core/csrc/ops/squeeze.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Squeeze::Squeeze(const torch::lazy::Value& input, int dim)
    : TsNode(torch::lazy::OpKind(at::aten::squeeze), {input},
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Squeeze::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Squeeze>(operands.at(0), dim_);
}

std::string Squeeze::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
