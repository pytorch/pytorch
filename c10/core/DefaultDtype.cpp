#include <c10/core/DefaultDtype.h>
#include <c10/util/typeid.h>

namespace c10 {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto default_dtype = caffe2::TypeMeta::Make<float>();
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto default_dtype_as_scalartype = default_dtype.toScalarType();
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto default_complex_dtype =
    caffe2::TypeMeta::Make<c10::complex<float>>();

void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = dtype;
  default_dtype_as_scalartype = default_dtype.toScalarType();
  switch (default_dtype_as_scalartype) {
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
ScalarType get_default_dtype_as_scalartype() {
  return default_dtype_as_scalartype;
}
const caffe2::TypeMeta get_default_complex_dtype() {
  return default_complex_dtype;
}
} // namespace c10
