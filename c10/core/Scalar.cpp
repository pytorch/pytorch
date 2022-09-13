#include <c10/core/Scalar.h>

namespace c10 {

Scalar Scalar::operator-() const {
  TORCH_CHECK(
      !isBoolean(),
      "torch boolean negative, the `-` operator, is not supported.");
  if (isFloatingPoint()) {
    return Scalar(-v.u.d);
  } else if (isComplex()) {
    return Scalar(-v.u.z);
  } else {
    return Scalar(-v.u.i);
  }
}

Scalar Scalar::conj() const {
  if (isComplex()) {
    return Scalar(std::conj(v.u.z));
  } else {
    return *this;
  }
}

Scalar Scalar::log() const {
  if (isComplex()) {
    return std::log(v.u.z);
  } else if (isFloatingPoint()) {
    return std::log(v.u.d);
  } else {
    return std::log(v.u.i);
  }
}

} // namespace c10
