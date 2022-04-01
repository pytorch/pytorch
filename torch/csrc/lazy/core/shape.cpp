#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/shape.h>

namespace torch {
namespace lazy {

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()) {}

std::string Shape::to_string() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

bool Shape::operator==(const Shape& other) const {
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.to_string();
}

size_t Shape::numel() const {
  size_t elts = 1;
  for (auto size : sizes_) {
    elts *= size;
  }
  return elts;
}

hash_t Shape::hash() const {
  return HashCombine(Hash(scalar_type_), DataHash(sizes_.data(), sizes_.size() * sizeof(int64_t)));
}

}  // namespace lazy
}  // namespace torch
