#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/ts_backend/view_ops/diagonal.h>

#include <cmath>

namespace torch {
namespace lazy {

const OpKind Diagonal::class_op_kind(at::aten::diagonal);

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

} // namespace lazy
} // namespace torch
