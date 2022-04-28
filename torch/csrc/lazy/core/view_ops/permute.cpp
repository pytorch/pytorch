#include <torch/csrc/lazy/core/view_ops/permute.h>

#include <torch/csrc/lazy/core/helpers.h>

namespace torch {
namespace lazy {

Permute::Permute(const Value& input, std::vector<int64_t> dims)
    : TsNode(
          OpKind(at::aten::permute),
          {input},
          /*num_outputs=*/1,
          MHash(dims)),
      dims_(std::move(dims)) {
  addComputedShape([&]() {
    return MakePermuteShape(operand(0).shape(), dims_);
  });
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dims=(" << c10::Join(", ", dims_) << ")";
  return ss.str();
}

Shape Permute::MakePermuteShape(
    const Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return Shape(
      source_shape.scalar_type(),
      PermuteDimensions(permutation, source_shape.sizes()));
}

} // namespace lazy
} // namespace torch
