#include "lazy_tensors/shape.h"

namespace lazy_tensors {

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()) {}

std::string Shape::ToString() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

bool Shape::operator==(const Shape& other) const {
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

}  // namespace lazy_tensors
