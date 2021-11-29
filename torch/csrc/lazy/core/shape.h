#pragma once

#include <ostream>
#include <vector>

#include <c10/core/Scalar.h>

namespace torch {
namespace lazy {

class TORCH_API Shape {
 public:
  Shape() = default;

  Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes);

  std::string to_string() const;

  c10::ScalarType scalar_type() const { return scalar_type_; }
  void set_scalar_type(at::ScalarType value) { scalar_type_ = value; }

  int64_t dim() const { return sizes_.size(); }
  c10::ArrayRef<int64_t> sizes() const { return sizes_; }
  int64_t size(int64_t dim) const { return sizes_.at(dim); }
  void set_size(int64_t dim, int64_t size) { sizes_.at(dim) = size; }

  bool operator==(const Shape& other) const;

 private:
  c10::ScalarType scalar_type_ {c10::ScalarType::Undefined};
  std::vector<int64_t> sizes_;
};

TORCH_API std::ostream& operator<<(std::ostream& out, const Shape& shape);

// TODO(alanwaketan): Rethink how code-gen uses shapes.
TORCH_API std::vector<Shape> convertShapes(
    const std::vector<at::ScalarType>& dtypes,
    const std::vector<std::vector<int64_t>>& shapes);

}  // namespace lazy
}  // namespace torch
