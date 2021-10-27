#include "lazy_tensor_core/csrc/shape_builder.h"

#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

ShapeBuilder& ShapeBuilder::Add(const lazy_tensors::Shape& shape, int64_t dim) {
  dims_.push_back({&shape, dim});
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(const lazy_tensors::Shape& shape,
                                c10::ArrayRef<int64_t> dimensions) {
  dims_.reserve(dimensions.size());
  for (auto dim : dimensions) {
    dims_.push_back({&shape, dim});
  }
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(int64_t size) {
  dims_.push_back({nullptr, size});
  return *this;
}

lazy_tensors::Shape ShapeBuilder::Build() const {
  std::vector<int64_t> dimensions;
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
  for (int64_t i = 0; i < shape.rank(); ++i) {
    const ShapeDim& sdim = dims_[i];
    if (sdim.shape != nullptr) {
      shape.set_dynamic_dimension(
          i, sdim.shape->is_dynamic_dimension(sdim.dim_or_size));
    }
  }
  return shape;
}

}  // namespace torch_lazy_tensors
