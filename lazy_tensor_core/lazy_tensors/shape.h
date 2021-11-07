#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include <c10/core/Scalar.h>
#include <c10/util/Logging.h>

namespace lazy_tensors {

// these helpers used to temporarily disable
// the "exact shape is not known" check for spots
// where we know it's actually save to get shapes
// for example, in EagerFallback, we should be able
// to always get the exact shape for arguments
// since we are materializing those arguments
// eventually, we will remove these helpers along
// with "the exact shape check"
void set_dynamic_mode_shape_check(bool v);
bool dynamic_mode_shape_check();

class Shape {
 public:
  Shape() = default;

  Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes);

  std::string ToString() const;

  c10::ScalarType scalar_type() const { return scalar_type_; }
  void set_scalar_type(at::ScalarType value) { scalar_type_ = value; }

  int64_t dim() const { return sizes_.size(); }
  c10::ArrayRef<int64_t> sizes() const {
    if (dynamic_mode_shape_check() && Shape::IsDynamicMode()) {
      throw std::runtime_error("Exact shape not known");
    }
    return sizes_;
  }
  int64_t size(int index) const {
    if (dynamic_mode_shape_check() && Shape::IsDynamicMode()) {
      throw std::runtime_error("Exact shape not known");
    }
    return sizes_.at(index);
  }
  void set_size(int index, int64_t value) { sizes_.at(index) = value; }

  bool operator==(const Shape& other) const;

  static bool IsDynamicMode();

  static void SetDynamicMode();

 private:
  c10::ScalarType scalar_type_ {c10::ScalarType::Undefined};
  std::vector<int64_t> sizes_;
  static std::atomic<bool> dynamic_mode_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);

// TODO(alanwaketan): Rethink how code-gen uses shapes.
std::vector<lazy_tensors::Shape> convertShapes(
    const std::vector<at::ScalarType>& dtypes,
    const std::vector<std::vector<int64_t>>& shapes);

}  // namespace lazy_tensors
