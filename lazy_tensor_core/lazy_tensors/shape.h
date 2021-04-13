#pragma once

#include <ostream>
#include <vector>

#include "absl/strings/str_join.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/layout.h"
#include "lazy_tensors/primitive_util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

class Shape {
 public:
  Shape() : element_type_(PrimitiveType::INVALID) {}

  Shape(PrimitiveType element_type, lazy_tensors::Span<const int64> dimensions)
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()) {}

  Shape(lazy_tensors::Span<const Shape> element_shapes)
      : element_type_(PrimitiveType::TUPLE),
        element_shapes_(element_shapes.begin(), element_shapes.end()) {}

  Shape(const client::ShapeData& shape_data)
      : element_type_(shape_data.element_type()),
        dimensions_(shape_data.dimensions()) {
    for (const client::ShapeData& element_shape : shape_data.element_shapes()) {
      element_shapes_.push_back(Shape(element_shape));
    }
    for (const int64_t dim_index : shape_data.minor_to_major()) {
      layout_.add_minor_to_major(dim_index);
    }
  }

  std::string ToString(bool print_layout = false) const {
    return absl::StrCat(PrimitiveTypeName(element_type_), "[",
                        absl::StrJoin(dimensions_, ","), "]");
  }

  int64 rank() const { return dimensions_.size(); }

  bool IsArray() const { return false; }

  bool IsTuple() const { return element_type_ == PrimitiveType::TUPLE; }

  bool is_dynamic_dimension(int dimension) const { return false; }

  void set_dynamic_dimension(int dimension, bool is_dynamic) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  lazy_tensors::Span<const bool> dynamic_dimensions() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void DeleteDimension(int64 dim_to_delete) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  PrimitiveType element_type() const { return element_type_; }
  void set_element_type(PrimitiveType value) { element_type_ = value; }

  int64 dimensions(int index) const {
    LTC_CHECK_LT(index, dimensions_.size());
    return dimensions_[index];
  }

  void set_dimensions(int index, int64 value) {
    LTC_CHECK_LT(index, dimensions_.size());
    dimensions_[index] = value;
  }

  lazy_tensors::Span<const int64> dimensions() const {
    return absl::MakeSpan(dimensions_);
  }

  int tuple_shapes_size() const { return element_shapes_.size(); }

  const Shape& tuple_shapes(int index) const {
    LTC_CHECK_GE(index, 0);
    LTC_CHECK_LT(index, element_shapes_.size());
    return element_shapes_[index];
  }
  const std::vector<Shape>& tuple_shapes() const { return element_shapes_; }

  const Layout& layout() const { return layout_; }

  Layout* mutable_layout() { return &layout_; }

  bool operator==(const Shape& other) const {
    return element_type_ == other.element_type_ &&
           dimensions_ == other.dimensions_;
  }

 private:
  PrimitiveType element_type_;
  std::vector<int64> dimensions_;
  std::vector<Shape> element_shapes_;
  Layout layout_;
};

class ProgramShape {
 public:
  ProgramShape(std::vector<Shape> parameters,
               std::vector<std::string> parameter_names, Shape result)
      : parameters_(std::move(parameters)),
        parameter_names_(std::move(parameter_names)),
        result_(std::move(result)) {
    LTC_CHECK_EQ(parameters_.size(), parameter_names_.size());
  }

  int parameters_size() const { return parameters_.size(); }

  const std::vector<Shape>& parameters() const { return parameters_; }

  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }

  const Shape& result() const { return result_; }

 private:
  std::vector<Shape> parameters_;
  std::vector<std::string> parameter_names_;
  Shape result_;
};

inline std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

inline client::ShapeData ToShapeData(const Shape& shape) {
  std::vector<client::ShapeData> element_shapes;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    element_shapes.push_back(ToShapeData(element_shape));
  }
  auto shape_dimensions = shape.dimensions();
  std::vector<int64_t> dimensions(shape_dimensions.begin(),
                                  shape_dimensions.end());
  auto minor_to_major = shape.layout().minor_to_major();
  return client::ShapeData(
      shape.element_type(), dimensions, element_shapes,
      std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end()));
}

}  // namespace lazy_tensors
