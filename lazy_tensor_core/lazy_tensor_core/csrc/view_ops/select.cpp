#include "lazy_tensor_core/csrc/view_ops/select.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Select::Select(const torch::lazy::Value& input, int64_t dim, int64_t start,
               int64_t end, int64_t stride)
    : BaseNode(
          select, {input},
          [&]() {
            return MakeSelectShape(torch::lazy::GetShapeFromTsValue(input), dim,
                                   start, end, stride);
          },
          /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << BaseNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

torch::lazy::Shape Select::MakeSelectShape(const torch::lazy::Shape& shape,
                                           int64_t dim, int64_t start,
                                           int64_t end, int64_t stride) {
  int64_t effective_stride = GetStride(start, end, stride);
  torch::lazy::Shape select_shape(shape);
  select_shape.set_size(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

int64_t Select::GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

SelectReverse::SelectReverse(const torch::lazy::Value& target,
                             const torch::lazy::Value& source, int64_t dim,
                             int64_t start, int64_t end, int64_t stride)
    : BaseNode(select_reverse, {target, source},
               {torch::lazy::GetShapeFromTsValue(target)},
               /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

std::string SelectReverse::ToString() const {
  std::stringstream ss;
  ss << BaseNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
