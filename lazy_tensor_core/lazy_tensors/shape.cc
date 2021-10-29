#include "lazy_tensors/shape.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
namespace lazy_tensors {

Shape::Shape(at::ScalarType element_type, c10::ArrayRef<int64_t> dimensions)
    : at_element_type_(element_type),
      dimensions_(dimensions.begin(), dimensions.end()) {}

Shape::Shape(const client::ShapeData& shape_data)
    : at_element_type_(
          torch_lazy_tensors::TensorTypeFromLtcType(shape_data.element_type())),
      dimensions_(shape_data.dimensions()) {
  for (const client::ShapeData& element_shape : shape_data.element_shapes()) {
    element_shapes_.push_back(Shape(element_shape));
  }
  for (const int64_t dim_index : shape_data.minor_to_major()) {
    layout_.add_minor_to_major(dim_index);
  }
}

void Shape::set_element_type(at::ScalarType value) {
  at_element_type_ = value;
}

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
