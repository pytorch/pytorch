#include "lazy_tensor_core/csrc/ops/select.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Select::Select(const torch::lazy::Value& input, int64_t dim, int64_t start,
               int64_t end, int64_t stride)
    : TsNode(
          ltc_select, {input},
          [&]() {
            return MakeSelectShape(ir::GetShapeFromTsValue(input), dim, start,
                                   end, stride);
          },
          /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Select::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Select>(operands.at(0), dim_, start_, end_, stride_);
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

lazy_tensors::Shape Select::MakeSelectShape(const lazy_tensors::Shape& shape,
                                            int64_t dim, int64_t start,
                                            int64_t end, int64_t stride) {
  int64_t effective_stride = GetStride(start, end, stride);
  lazy_tensors::Shape select_shape(shape);
  select_shape.set_dimensions(
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

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
