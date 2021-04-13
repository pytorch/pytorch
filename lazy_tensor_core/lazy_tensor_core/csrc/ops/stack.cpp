#include "lazy_tensor_core/csrc/ops/stack.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Stack::Stack(lazy_tensors::Span<const ir::Value> values,
             lazy_tensors::int64 dim)
    : Node(ir::OpKind(at::aten::stack), values,
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Stack::Clone(OpList operands) const {
  return MakeNode<Stack>(operands, dim_);
}

std::string Stack::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
