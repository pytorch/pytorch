#include "lazy_tensors/shape.h"

namespace lazy_tensors {

bool thread_local shape_check = true;
void set_dynamic_mode_shape_check(bool v) {
  shape_check = v;
}
bool dynamic_mode_shape_check() {
  return shape_check;
}

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()) {}

std::string Shape::ToString() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

bool Shape::IsDynamicMode() { return dynamic_mode_.load(); }

void Shape::SetDynamicMode() { dynamic_mode_ = true; }

std::atomic<bool> Shape::dynamic_mode_{false};

bool Shape::operator==(const Shape& other) const {
  if (dynamic_mode_shape_check() && Shape::IsDynamicMode()) {
      throw std::runtime_error("Exact shape not known");
  }
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

std::vector<lazy_tensors::Shape> convertShapes(
    const std::vector<at::ScalarType>& dtypes,
    const std::vector<std::vector<int64_t>>& shapes) {
  TORCH_INTERNAL_ASSERT(dtypes.size() == shapes.size());

  std::vector<lazy_tensors::Shape> shape;
  for (int i = 0; i < dtypes.size(); i++) {
    shape.emplace_back(dtypes[i], shapes[i]);
  }

  return shape;
}

}  // namespace lazy_tensors
