#include "lazy_tensor_core/csrc/ops/diagonal.h"

#include <cmath>

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Diagonal::Diagonal(const Value& input, lazy_tensors::int64 offset,
                   lazy_tensors::int64 dim1, lazy_tensors::int64 dim2)
    : Node(ir::OpKind(at::aten::diagonal), {input},
           [&]() {
             return MakeDiagonalShape(input.shape(), offset, dim1, dim2);
           },
           /*num_outputs=*/1, lazy_tensors::util::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

NodePtr Diagonal::Clone(OpList operands) const {
  return MakeNode<Diagonal>(operands.at(0), offset_, dim1_, dim2_);
}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

lazy_tensors::Shape Diagonal::MakeDiagonalShape(
    const lazy_tensors::Shape& shape, lazy_tensors::int64 offset,
    lazy_tensors::int64 dim1, lazy_tensors::int64 dim2) {
  std::vector<lazy_tensors::int64> dimensions;
  for (lazy_tensors::int64 dim = 0; dim < shape.rank(); ++dim) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.dimensions(dim));
    }
  }
  lazy_tensors::int64 dsize;
  if (offset >= 0) {
    dsize = std::max<lazy_tensors::int64>(
        std::min(shape.dimensions(dim1), shape.dimensions(dim2) - offset), 0);
  } else {
    dsize = std::max<lazy_tensors::int64>(
        std::min(shape.dimensions(dim1) + offset, shape.dimensions(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return lazy_tensors::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
