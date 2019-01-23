#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

namespace at {

// A variant abstraction for things with a ScalarType, allowing functions like
// resultType to work with scalar/tensor/type at the same time.
//
// e.g.
//   int f(ScalarTypeSource... args);
//   result = f(tensor1, tensor2, scalar, 3.0)
//
// Supports the following types:
// - ScalarType: Yields the ScalarType itself
// - bool: Byte
// - Scalar: Long, Double or ComplexDouble (Python scalar types)
// - Tensor: the dtype of the tensor
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
