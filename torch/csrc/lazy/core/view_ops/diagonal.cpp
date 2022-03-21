#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/view_ops/diagonal.h>

#include <cmath>

namespace torch {
namespace lazy {

Diagonal::Diagonal(
    const Value& input,
    int64_t offset,
    int64_t dim1,
    int64_t dim2)
    : TsNode(
          OpKind(at::aten::diagonal),
          {input},
          [&]() {
            return MakeDiagonalShape(input.shape(), offset, dim1, dim2);
          },
          /*num_outputs=*/1,
          MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

Shape Diagonal::MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  std::vector<int64_t> dimensions;
  for (const auto dim : c10::irange(shape.dim())) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.size(dim));
    }
  }
  int64_t dsize = 0;
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1), shape.size(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1) + offset, shape.size(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return Shape(shape.scalar_type(), dimensions);
}

} // namespace lazy
} // namespace torch
