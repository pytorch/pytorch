#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

namespace at {

// Holder for an object that contributes scalar type to an operation,
// e.g. resultType.
class ScalarTypeSource final {
 public:
  ScalarTypeSource()
      : is_scalar_type_(false), is_scalar_(false), is_tensor_(false) {}
  /* implicit */ ScalarTypeSource(ScalarType scalar_type)
      : is_scalar_type_(true),
        is_scalar_(false),
        is_tensor_(false),
        scalar_type_(scalar_type) {}
  /* implicit */ ScalarTypeSource(Scalar scalar)
      : is_scalar_type_(false),
        is_scalar_(true),
        is_tensor_(false),
        scalar_(scalar) {}
  /* implicit */ ScalarTypeSource(Tensor tensor)
      : is_scalar_type_(false),
        is_scalar_(false),
        is_tensor_(true),
        tensor_(tensor) {}

  bool isScalarType() {
    return is_scalar_type_;
  }
  bool isScalar() {
    return is_scalar_;
  }
  bool isTensor() {
    return is_tensor_;
  }

  ScalarType asScalarType() {
    AT_CHECK(isScalarType());
    return scalar_type_;
  }
  Scalar asScalar() {
    AT_CHECK(isScalar());
    return scalar_;
  }
  Tensor asTensor() {
    AT_CHECK(isTensor());
    return tensor_;
  }

 private:
  bool is_scalar_type_;
  bool is_scalar_;
  bool is_tensor_;
  Scalar scalar_;
  ScalarType scalar_type_;
  Tensor tensor_;
};

} // namespace at
