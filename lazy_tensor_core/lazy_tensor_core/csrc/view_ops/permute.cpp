#include "lazy_tensor_core/csrc/view_ops/permute.h"

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Permute::Permute(const torch::lazy::Value& input, std::vector<int64_t> dims)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::permute), {input},
                          /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {
  SetShapeDeferred([&]() { return compiler::InferShape(this); });
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dims=(" << c10::Join(", ", dims_)
     << ")";
  return ss.str();
}

torch::lazy::Shape Permute::MakePermuteShape(
    const torch::lazy::Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return torch::lazy::Shape(
      source_shape.scalar_type(),
      torch::lazy::Permute(permutation, source_shape.sizes()));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
