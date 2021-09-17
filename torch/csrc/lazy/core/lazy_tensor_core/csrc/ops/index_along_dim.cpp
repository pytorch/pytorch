#include "lazy_tensor_core/csrc/ops/index_along_dim.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexAlongDim::IndexAlongDim(OpKind op, const ir::Value& buffer,
                             const ir::Value& index, const ir::Value& value,
                             lazy_tensors::int64 dim)
    : Node(op, {buffer, index, value},
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

std::string IndexAlongDim::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

NodePtr IndexAlongDim::Clone(OpList operands) const {
  return MakeNode<IndexAlongDim>(op(), operands.at(0), operands.at(1),
                                 operands.at(2), dim_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
