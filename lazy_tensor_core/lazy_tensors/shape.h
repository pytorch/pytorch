#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include <c10/core/Scalar.h>
#include <c10/util/Logging.h>

namespace lazy_tensors {

class Shape {
 public:
  Shape() = default;

  Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes);

  std::string ToString() const;

  c10::ScalarType scalar_type() const { return scalar_type_; }
  void set_scalar_type(at::ScalarType value) { scalar_type_ = value; }

  int64_t dim() const { return sizes_.size(); }
  c10::ArrayRef<int64_t> sizes() const { return sizes_; }
  int64_t size(int index) const { return sizes_.at(index); }
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
