#include <torch/csrc/lazy/core/view_ops/narrow.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

Narrow::Narrow(
    const Value& input,
    c10::ArrayRef<int64_t> base_indices,
    c10::ArrayRef<int64_t> sizes)
    : TsNode(
          OpKind(at::aten::narrow),
          {input},
          /*num_outputs=*/1,
          MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {
  addComputedShape([&]() {
    return Shape(operand(0).shape().scalar_type(), sizes);
  });
}

std::string Narrow::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << "), sizes=("
     << c10::Join(", ", sizes_) << ")";
  return ss.str();
}

} // namespace lazy
} // namespace torch
