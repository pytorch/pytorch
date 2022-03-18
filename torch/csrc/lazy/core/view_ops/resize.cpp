#include <torch/csrc/lazy/core/view_ops/resize.h>

namespace torch {
namespace lazy {

namespace {

Shape NodeOutputShape(const Value& input, c10::ArrayRef<int64_t> size) {
  return Shape(input.shape().scalar_type(), size);
}

} // namespace

Resize::Resize(const Value& input, std::vector<int64_t> size)
    : TsNode(
          OpKind(at::aten::resize),
          {input},
          [&]() { return NodeOutputShape(input, size); },
          /*num_outputs=*/1,
          MHash(size)),
      size_(std::move(size)) {}

std::string Resize::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_) << ")";
  return ss.str();
}

} // namespace lazy
} // namespace torch
