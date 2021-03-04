#include <c10/core/Scalar.h>

namespace c10 {

Scalar Scalar::operator-() const {
  TORCH_CHECK(!isBoolean(), "torch boolean negative, the `-` operator, is not supported.");
  if (isFloatingPoint()) {
    return Scalar(-v.d);
  } else if (isComplex()) {
    return Scalar(-(*v.z).val);
  } else {
    return Scalar(-v.i);
  }
}

Scalar Scalar::conj() const {
  if (isComplex()) {
    return Scalar(std::conj((*v.z).val));
  } else {
    return *this;
  }
}

Scalar Scalar::log() const {
  if (isComplex()) {
    return std::log((*v.z).val);
  } else if (isFloatingPoint()) {
    return std::log(v.d);
  } else {
    return std::log(v.i);
  }
}

}  // namespace c10
