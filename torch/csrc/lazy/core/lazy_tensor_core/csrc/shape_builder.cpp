#include "lazy_tensor_core/csrc/shape_builder.h"

#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

ShapeBuilder& ShapeBuilder::Add(const lazy_tensors::Shape& shape,
                                lazy_tensors::int64 dim) {
  dims_.push_back({&shape, dim});
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(
    const lazy_tensors::Shape& shape,
    lazy_tensors::Span<const lazy_tensors::int64> dimensions) {
  dims_.reserve(dimensions.size());
  for (auto dim : dimensions) {
    dims_.push_back({&shape, dim});
  }
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(lazy_tensors::int64 size) {
  dims_.push_back({nullptr, size});
  return *this;
}

lazy_tensors::Shape ShapeBuilder::Build() const {
  std::vector<lazy_tensors::int64> dimensions;
  dimensions.reserve(dims_.size());
  for (auto& sdim : dims_) {
    if (sdim.shape != nullptr) {
      dimensions.push_back(sdim.shape->dimensions(sdim.dim_or_size));
    } else {
      dimensions.push_back(sdim.dim_or_size);
    }
  }
  lazy_tensors::Shape shape =
      lazy_tensors::ShapeUtil::MakeShape(type_, dimensions);
  for (lazy_tensors::int64 i = 0; i < shape.rank(); ++i) {
    const ShapeDim& sdim = dims_[i];
    if (sdim.shape != nullptr) {
      shape.set_dynamic_dimension(
          i, sdim.shape->is_dynamic_dimension(sdim.dim_or_size));
    }
  }
  return shape;
}

}  // namespace torch_lazy_tensors
