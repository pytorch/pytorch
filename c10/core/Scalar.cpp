#include <c10/core/Scalar.h>

namespace c10 {

Scalar Scalar::operator-() const {
  TORCH_CHECK(!isBoolean(), "torch boolean negative, the `-` operator, is not suppported.");
  if (isFloatingPoint()) {
    return Scalar(-v.d);
  } else if (isComplex()) {
    return Scalar(-v.z);
  } else {
    return Scalar(-v.i);
  }
}

Scalar Scalar::conj() const {
  if (isComplex()) {
    return Scalar(std::conj(v.z));
  } else {
    return *this;
  }
}

}  // namespace c10
