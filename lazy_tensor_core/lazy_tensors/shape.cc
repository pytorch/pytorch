#include "lazy_tensors/shape.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
namespace lazy_tensors {

Shape::Shape(at::ScalarType element_type, c10::ArrayRef<int64_t> dimensions)
    : at_element_type_(element_type),
      dimensions_(dimensions.begin(), dimensions.end()) {}

void Shape::set_element_type(at::ScalarType value) {
  at_element_type_ = value;
}


}  // namespace lazy_tensors
