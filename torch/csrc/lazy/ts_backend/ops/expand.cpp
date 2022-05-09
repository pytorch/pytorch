#include <torch/csrc/lazy/ts_backend/ops/expand.h>

namespace torch {
namespace lazy {

Expand::Expand(
    const Value& input,
    std::vector<int64_t> size,
    bool is_scalar_expand)
    : TsNode(
          OpKind(at::aten::expand),
          {input},
          /*num_outputs=*/1,
          MHash(size, is_scalar_expand)),
      size_(std::move(size)),
      is_scalar_expand_(is_scalar_expand) {
  addComputedShape(
      [&]() { return Shape(input.shape().scalar_type(), size_); });
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), is_scalar_expand=" << is_scalar_expand_;
  return ss.str();
}

} // namespace lazy
} // namespace torch
