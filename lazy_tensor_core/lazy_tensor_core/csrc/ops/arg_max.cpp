#include "lazy_tensor_core/csrc/ops/arg_max.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ArgMax::ArgMax(const torch::lazy::Value& input, lazy_tensors::int64 dim, bool keepdim)
    : TsNode(torch::lazy::OpKind(at::aten::argmax), {input},
           /*num_outputs=*/1, torch::lazy::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ArgMax::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ArgMax>(operands.at(0), dim_, keepdim_);
}

std::string ArgMax::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
