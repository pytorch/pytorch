#include "ATen/core/Scalar.h"

namespace at {

Scalar Scalar::operator-() const {
 if (isFloatingPoint()) {
   return Scalar(-v.d);
 } else {
   return Scalar(-v.i);
 }
}

}  // namespace at
