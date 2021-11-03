#include "lazy_tensors/shape.h"

namespace lazy_tensors {

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()) {}

}  // namespace lazy_tensors
