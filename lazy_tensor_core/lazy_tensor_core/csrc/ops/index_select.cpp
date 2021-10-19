#include "lazy_tensor_core/csrc/ops/index_select.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexSelect::IndexSelect(const torch::lazy::Value& input, lazy_tensors::int64 dim,
                         const torch::lazy::Value& index)
    : TsNode(torch::lazy::OpKind(at::aten::index_select), {input, index},
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr IndexSelect::Clone(OpList operands) const {
  return torch::lazy::MakeNode<IndexSelect>(operands.at(0), dim_, operands.at(1));
}

std::string IndexSelect::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
