#include <torch/csrc/lazy/core/view_ops/diagonal_view_update.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

DiagonalViewUpdate::DiagonalViewUpdate(
    const Value& target,
    const Value& input,
    int64_t offset,
    int64_t dim1,
    int64_t dim2)
    : TsNode(
          ltc_diagonal_view_update,
          {target, input},
          {target.shape()},
          /*num_outputs=*/1,
          MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

} // namespace lazy
} // namespace torch
