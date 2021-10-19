#include "lazy_tensor_core/csrc/ops/generic_slice.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

GenericSlice::GenericSlice(
    const torch::lazy::Value& input,
    lazy_tensors::Span<const lazy_tensors::int64> base_indices,
    lazy_tensors::Span<const lazy_tensors::int64> sizes)
    : TsNode(ltc_generic_slice, {input},
           /*num_outputs=*/1, torch::lazy::MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr GenericSlice::Clone(OpList operands) const {
  return torch::lazy::MakeNode<GenericSlice>(operands.at(0), base_indices_, sizes_);
}

std::string GenericSlice::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base_indices=("
     << lazy_tensors::StrJoin(base_indices_, ", ") << "), sizes=("
     << lazy_tensors::StrJoin(sizes_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
