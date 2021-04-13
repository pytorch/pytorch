#include "lazy_tensor_core/csrc/ops/select.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Select::Select(const Value& input, lazy_tensors::int64 dim,
               lazy_tensors::int64 start, lazy_tensors::int64 end,
               lazy_tensors::int64 stride)
    : Node(ltc_select, {input},
           [&]() {
             return MakeSelectShape(input.shape(), dim, start, end, stride);
           },
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Select::Clone(OpList operands) const {
  return MakeNode<Select>(operands.at(0), dim_, start_, end_, stride_);
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

lazy_tensors::Shape Select::MakeSelectShape(const lazy_tensors::Shape& shape,
                                            lazy_tensors::int64 dim,
                                            lazy_tensors::int64 start,
                                            lazy_tensors::int64 end,
                                            lazy_tensors::int64 stride) {
  lazy_tensors::int64 effective_stride = GetStride(start, end, stride);
  lazy_tensors::Shape select_shape(shape);
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

lazy_tensors::int64 Select::GetStride(lazy_tensors::int64 start,
                                      lazy_tensors::int64 end,
                                      lazy_tensors::int64 stride) {
  if (stride == 0) {
    LTC_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
