#include <c10/core/Scalar.h>

namespace c10 {

Scalar Scalar::operator-() const {
  TORCH_CHECK(
      !isBoolean(),
      "torch boolean negative, the `-` operator, is not supported.");
  if (isFloatingPoint()) {
    TORCH_CHECK(!isSymbolic(), "NYI negate symbolic float");
    return Scalar(-v.d);
  } else if (isComplex()) {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    return Scalar(-v.z);
  } else if (isIntegral(false)) {
    TORCH_CHECK(!isSymbolic(), "NYI negate symbolic int");
    return Scalar(-v.i);
  }
  TORCH_INTERNAL_ASSERT(false, "unknown ivalue tag ", static_cast<int>(tag));
}

Scalar Scalar::conj() const {
  if (isComplex()) {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    return Scalar(std::conj(v.z));
  } else {
    return *this;
  }
}

Scalar Scalar::log() const {
  if (isComplex()) {
    TORCH_INTERNAL_ASSERT(!isSymbolic());
    return std::log(v.z);
  } else if (isFloatingPoint()) {
    TORCH_CHECK(!isSymbolic(), "NYI log symbolic float");
    return std::log(v.d);
  } else if (isIntegral(false)) {
    TORCH_CHECK(!isSymbolic(), "NYI log symbolic int");
    return std::log(v.i);
  }
  TORCH_INTERNAL_ASSERT(false, "unknown ivalue tag ", static_cast<int>(tag));
}

} // namespace c10
