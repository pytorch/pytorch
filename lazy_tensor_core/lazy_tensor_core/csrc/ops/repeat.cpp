#include "lazy_tensor_core/csrc/ops/repeat.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Repeat::Repeat(const Value& input, std::vector<lazy_tensors::int64> repeats)
    : TsNode(ir::OpKind(at::aten::repeat), {input},
           /*num_outputs=*/1, torch::lazy::MHash(repeats)),
      repeats_(std::move(repeats)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Repeat::Clone(OpList operands) const {
  return MakeNode<Repeat>(operands.at(0), repeats_);
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", repeats=("
     << lazy_tensors::StrJoin(repeats_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
