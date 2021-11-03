#include "lazy_tensors/shape.h"

namespace lazy_tensors {

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> dimensions)
    : scalar_type_(scalar_type),
      dimensions_(dimensions.begin(), dimensions.end()) {}

}  // namespace lazy_tensors
