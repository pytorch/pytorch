#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/ts_backend/view_ops/select.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

const OpKind Select::class_op_kind(at::aten::select);

Select::Select(
    const Value& input,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride)
    : TsNode(
          OpKind(at::aten::select),
          {input},
          [&]() {
            return MakeSelectShape(input.shape(), dim, start, end, stride);
          },
          /*num_outputs=*/1,
          MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

int64_t Select::GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

} // namespace lazy
} // namespace torch
