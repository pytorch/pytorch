#include <c10/util/typeid.h>
#include <c10/core/DefaultDtype.h>

namespace c10 {
static caffe2::TypeMeta default_dtype = caffe2::TypeMeta::Make<float>();
static caffe2::TypeMeta default_complex_dtype = caffe2::TypeMeta::Make<c10::complex<float>>();

void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = dtype;
  switch (default_dtype.toScalarType()) {
    case ScalarType::Half:
      default_complex_dtype = ScalarType::ComplexHalf;
      break;
    case ScalarType::Double:
      default_complex_dtype = ScalarType::ComplexDouble;
      break;
    default:
      default_complex_dtype = ScalarType::ComplexFloat;
      break;
  }
}

const caffe2::TypeMeta get_default_dtype() {
  return default_dtype;
}
const caffe2::TypeMeta get_default_complex_dtype() {
  return default_complex_dtype;
}
} // namespace c10
