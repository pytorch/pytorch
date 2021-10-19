#include "lazy_tensor_core/csrc/ops/update_slice.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpdateSlice::UpdateSlice(
    const torch::lazy::Value& input, const torch::lazy::Value& source,
    lazy_tensors::Span<const lazy_tensors::int64> base_indices)
    : TsNode(ltc_update_slice, {input, source},
           /*num_outputs=*/1, torch::lazy::MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpdateSlice::Clone(OpList operands) const {
  return torch::lazy::MakeNode<UpdateSlice>(operands.at(0), operands.at(1), base_indices_);
}

std::string UpdateSlice::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base_indices=("
     << lazy_tensors::StrJoin(base_indices_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
