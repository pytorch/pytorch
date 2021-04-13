#include "lazy_tensor_core/csrc/ops/max_in_dim.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaxInDim::MaxInDim(const Value& input, lazy_tensors::int64 dim, bool keepdim)
    : Node(ir::OpKind(at::aten::max), {input},
           /*num_outputs=*/2, lazy_tensors::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MaxInDim::Clone(OpList operands) const {
  return MakeNode<MaxInDim>(operands.at(0), dim_, keepdim_);
}

std::string MaxInDim::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
