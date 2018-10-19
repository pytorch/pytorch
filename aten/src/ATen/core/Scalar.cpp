#include "ATen/core/Scalar.h"

namespace at {

Scalar Scalar::operator-() const {
  if (isFloatingPoint()) {
    return Scalar(-v.d);
  } else if (isComplex()) {
    return Scalar(std::complex<double>(-v.z[0], -v.z[1]));
  } else {
    return Scalar(-v.i);
  }
}

}  // namespace at
