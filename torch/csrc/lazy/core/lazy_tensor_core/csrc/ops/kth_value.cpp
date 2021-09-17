#include "lazy_tensor_core/csrc/ops/kth_value.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

KthValue::KthValue(const Value& input, lazy_tensors::int64 k,
                   lazy_tensors::int64 dim, bool keepdim)
    : Node(ir::OpKind(at::aten::kthvalue), {input},
           /*num_outputs=*/2, lazy_tensors::util::MHash(k, dim, keepdim)),
      k_(k),
      dim_(dim),
      keepdim_(keepdim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr KthValue::Clone(OpList operands) const {
  return MakeNode<KthValue>(operands.at(0), k_, dim_, keepdim_);
}

std::string KthValue::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
