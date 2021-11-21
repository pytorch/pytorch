#include "lazy_tensor_core/csrc/view_ops/diagonal.h"

#include <cmath>

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Diagonal::Diagonal(const torch::lazy::Value& input, int64_t offset,
                   int64_t dim1, int64_t dim2)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::diagonal), {input},
          [&]() {
            return MakeDiagonalShape(torch::lazy::GetShapeFromTsValue(input),
                                     offset, dim1, dim2);
          },
          /*num_outputs=*/1, torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", offset=" << offset_
     << ", dim1=" << dim1_ << ", dim2=" << dim2_;
  return ss.str();
}

torch::lazy::Shape Diagonal::MakeDiagonalShape(const torch::lazy::Shape& shape,
                                               int64_t offset, int64_t dim1,
                                               int64_t dim2) {
  std::vector<int64_t> dimensions;
  for (int64_t dim = 0; dim < shape.dim(); ++dim) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.size(dim));
    }
  }
  int64_t dsize;
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1), shape.size(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1) + offset, shape.size(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return torch::lazy::Shape(shape.scalar_type(), dimensions);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
