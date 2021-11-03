#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include <c10/core/Scalar.h>
#include <c10/util/Logging.h>

namespace lazy_tensors {

class Shape {
 public:
  Shape() : scalar_type_(c10::ScalarType::Undefined) {}

  Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes);

  Shape(c10::ArrayRef<Shape> element_shapes)
      : is_tuple_(true),
        scalar_type_(c10::ScalarType::Undefined),
        element_shapes_(element_shapes.begin(), element_shapes.end()) {
    CHECK(element_shapes.size() > 0);
    // TODO(whc) it's not really clear what the definition of element shape
    // should be for a tuple shape.  However, for tuple shapes, we appear
    // to be accessing the scalar_type field in some places.  Fix this.
    scalar_type_ = element_shapes[0].scalar_type();
  }

  std::string ToString(bool print_layout = false) const {
    return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
  }

  c10::ScalarType scalar_type() const { return scalar_type_; }
  void set_scalar_type(at::ScalarType value) { scalar_type_ = value; }

  int64_t dim() const { return sizes_.size(); }

  int64_t size(int index) const {
    CHECK_LT(index, sizes_.size());
    return sizes_[index];
  }

  c10::ArrayRef<int64_t> sizes() const { return sizes_; }

  void set_size(int index, int64_t value) {
    CHECK_LT(index, sizes_.size());
    sizes_[index] = value;
  }

  // TODO(whc) remove tuple support? or keep it (But make sizes() methods
  // work consistently with it somehow?)
  bool IsTuple() const { return is_tuple_; }
  int tuple_shapes_size() const { return element_shapes_.size(); }

  const Shape& tuple_shapes(int index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, element_shapes_.size());
    return element_shapes_[index];
  }
  const std::vector<Shape>& tuple_shapes() const { return element_shapes_; }

  bool operator==(const Shape& other) const {
    return scalar_type_ == other.scalar_type_ &&
           sizes_ == other.sizes_;
  }

 private:
  bool is_tuple_ = false;
  c10::ScalarType scalar_type_;
  std::vector<int64_t> sizes_;
  std::vector<Shape> element_shapes_;
};

inline std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

}  // namespace lazy_tensors
