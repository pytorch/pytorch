#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

namespace at {

// Holder for an object that contributes scalar type to an operation,
// e.g. resultType.
class ScalarTypeSource final {
 public:
  ScalarTypeSource() {}
  /* implicit */ ScalarTypeSource(ScalarType scalar_type)
      : is_scalar_type_(true), scalar_type_(scalar_type) {}
  /* implicit */ ScalarTypeSource(bool boolean)
      : is_scalar_(true), scalar_type_(ScalarType::Byte) {}
  /* implicit */ ScalarTypeSource(Scalar scalar)
      : is_scalar_(true), scalar_type_(scalar.scalarType()) {}
  /* implicit */ ScalarTypeSource(Tensor tensor)
      : is_tensor_(true), scalar_type_(tensor.scalar_type()),
        zero_dim_(tensor.dim() == 0) {}

  bool isScalarType() const {
    return is_scalar_type_;
  }

  bool isScalar() const {
    return is_scalar_;
  }

  bool isTensor() const {
    return is_tensor_;
  }

  bool isZeroDimTensor() const {
    return is_tensor_ && zero_dim_;
  }

  bool isNonZeroDimTensor() const {
    return is_tensor_ && !zero_dim_;
  }

  ScalarType scalarType() const {
    return scalar_type_;
  }

 private:
  const bool is_scalar_type_ = false;
  const bool is_scalar_ = false;
  const bool is_tensor_ = false;
  const ScalarType scalar_type_ = ScalarType::Undefined;
  const bool zero_dim_ = false;
};

} // namespace at
