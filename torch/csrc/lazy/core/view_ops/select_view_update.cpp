#include <torch/csrc/lazy/core/view_ops/select_view_update.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/select.h>

namespace torch {
namespace lazy {

SelectViewUpdate::SelectViewUpdate(
    const Value& target,
    const Value& source,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride)
    : TsNode(
          ltc_select_view_update,
          {target, source},
          {target.shape()},
          /*num_outputs=*/1,
          MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

std::string SelectViewUpdate::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

} // namespace lazy
} // namespace torch
