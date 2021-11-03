#include "lazy_tensor_core/csrc/ops/diagonal.h"

#include <cmath>

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Diagonal::Diagonal(const torch::lazy::Value& input, int64_t offset,
                   int64_t dim1, int64_t dim2)
    : TsNode(
          torch::lazy::OpKind(at::aten::diagonal), {input},
          [&]() {
            return MakeDiagonalShape(ir::GetShapeFromTsValue(input), offset,
                                     dim1, dim2);
          },
          /*num_outputs=*/1, torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

lazy_tensors::Shape Diagonal::MakeDiagonalShape(
    const lazy_tensors::Shape& shape, int64_t offset, int64_t dim1,
    int64_t dim2) {
  std::vector<int64_t> dimensions;
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.dimensions(dim));
    }
  }
  int64_t dsize;
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.dimensions(dim1), shape.dimensions(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.dimensions(dim1) + offset, shape.dimensions(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return lazy_tensors::ShapeUtil::MakeShape(shape.scalar_type(), dimensions);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
