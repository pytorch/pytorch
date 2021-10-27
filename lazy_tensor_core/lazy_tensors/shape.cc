#include "lazy_tensors/shape.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
namespace lazy_tensors {

Shape::Shape(at::ScalarType element_type, c10::ArrayRef<int64_t> dimensions)
    : at_element_type_(element_type),
      dimensions_(dimensions.begin(), dimensions.end()),
      dynamic_dimensions_(dimensions.size(), false) {}

Shape::Shape(const client::ShapeData& shape_data)
    : at_element_type_(
          torch_lazy_tensors::TensorTypeFromLtcType(shape_data.element_type())),
      dimensions_(shape_data.dimensions()),
      dynamic_dimensions_(shape_data.dimensions().size(), false) {
  for (const client::ShapeData& element_shape : shape_data.element_shapes()) {
    element_shapes_.push_back(Shape(element_shape));
  }
  for (const int64_t dim_index : shape_data.minor_to_major()) {
    layout_.add_minor_to_major(dim_index);
  }
}

void Shape::DeleteDimension(int64_t dim_to_delete) {
  CHECK(IsArray());
  CHECK_GE(dim_to_delete, 0);
  CHECK_LT(dim_to_delete, dimensions_.size());
  dimensions_.erase(dimensions_.begin() + dim_to_delete);
  for (int64_t i = 0; i < layout_.minor_to_major().size();) {
    if (layout_.minor_to_major(i) == dim_to_delete) {
      layout_.mutable_minor_to_major()->erase(
          layout_.mutable_minor_to_major()->begin() + i);
      continue;
    }
    if (layout_.minor_to_major(i) > dim_to_delete) {
      (*layout_.mutable_minor_to_major())[i] -= 1;
    }
    ++i;
  }
}

void Shape::set_element_type(at::ScalarType value) {
  at_element_type_ = value;
}

bool Shape::IsDynamicMode() { return dynamic_mode_.load(); }

void Shape::SetDynamicMode() { dynamic_mode_ = true; }

std::atomic<bool> Shape::dynamic_mode_{false};


client::ShapeData ToShapeData(const Shape& shape) {
  std::vector<client::ShapeData> element_shapes;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    element_shapes.push_back(ToShapeData(element_shape));
  }
  auto shape_dimensions = shape.dimensions();
  std::vector<int64_t> dimensions(shape_dimensions.begin(),
                                  shape_dimensions.end());
  auto minor_to_major = shape.layout().minor_to_major();
  return client::ShapeData(
      torch_lazy_tensors::MakeLtcPrimitiveType(shape.at_element_type(),
                                               nullptr),
      dimensions, element_shapes,
      std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end()));
}

}  // namespace lazy_tensors
