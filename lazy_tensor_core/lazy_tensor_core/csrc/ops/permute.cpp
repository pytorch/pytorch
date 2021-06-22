#include "lazy_tensor_core/csrc/ops/permute.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Permute::Permute(const Value& input, std::vector<lazy_tensors::int64> dims)
    : Node(ir::OpKind(at::aten::permute), {input},
           /*num_outputs=*/1, lazy_tensors::util::MHash(dims)),
      dims_(std::move(dims)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Permute::Clone(OpList operands) const {
  return MakeNode<Permute>(operands.at(0), dims_);
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=(" << lazy_tensors::StrJoin(dims_, ", ")
     << ")";
  return ss.str();
}

lazy_tensors::Shape Permute::MakePermuteShape(
    const lazy_tensors::Shape& source_shape,
    lazy_tensors::Span<const lazy_tensors::int64> permutation) {
  return Helpers::GetDynamicReshape(
      source_shape, Helpers::Permute(permutation, source_shape.dimensions()));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
