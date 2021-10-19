#include "lazy_tensor_core/csrc/ops/index_along_dim.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexAlongDim::IndexAlongDim(OpKind op, const torch::lazy::Value& buffer,
                             const torch::lazy::Value& index, const torch::lazy::Value& value,
                             lazy_tensors::int64 dim)
    : TsNode(op, {buffer, index, value},
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

std::string IndexAlongDim::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

NodePtr IndexAlongDim::Clone(OpList operands) const {
  return torch::lazy::MakeNode<IndexAlongDim>(op(), operands.at(0), operands.at(1),
                                 operands.at(2), dim_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
