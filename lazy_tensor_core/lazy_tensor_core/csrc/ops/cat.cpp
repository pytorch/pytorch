#include "lazy_tensor_core/csrc/ops/cat.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Cat::Cat(OpList values, lazy_tensors::int64 dim)
    : TsNode(torch::lazy::OpKind(at::aten::cat), values,
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Cat::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Cat>(operands, dim_);
}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
