#include "lazy_tensor_core/csrc/ops/permute.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Permute::Permute(const torch::lazy::Value& input, std::vector<int64_t> dims)
    : TsNode(torch::lazy::OpKind(at::aten::permute), {input},
             /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Permute::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Permute>(operands.at(0), dims_);
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dims=(" << c10::Join(", ", dims_) << ")";
  return ss.str();
}

lazy_tensors::Shape Permute::MakePermuteShape(
    const lazy_tensors::Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return lazy_tensors::ShapeUtil::MakeShape(
      source_shape.at_element_type(),
      Helpers::Permute(permutation, source_shape.dimensions()));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
