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

template<typename T>
bool Scalar::equal(T num) const {
  if (isComplex()) {
    return v.z == num;
  } else if (isFloatingPoint()) {
    return v.d == num;
  } else {
    return v.i == num;
  }
}

}  // namespace c10
