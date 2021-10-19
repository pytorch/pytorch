#include "lazy_tensor_core/csrc/ops/amax.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Amax::Amax(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> dimensions,
           bool keepdim)
    : TsNode(torch::lazy::OpKind(at::aten::amax), {input},
           /*num_outputs=*/1, torch::lazy::MHash(dimensions, keepdim)),
      dimensions_(std::move(dimensions)),
      keepdim_(keepdim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Amax::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Amax>(operands.at(0), dimensions_, keepdim_);
}

std::string Amax::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString()
     << ", dimensions=" << lazy_tensors::StrJoin(dimensions_, ", ")
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
