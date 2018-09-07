#pragma once

#include "ATen/core/Scalar.h"
#include "ATen/Tensor.h"

namespace at {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no way
// to implement this without going through Derived Types (which are not part of core).
inline Tensor scalar_to_tensor(Scalar s) {
  if (s.isFloatingPoint()) {
    return CPU(kDouble).scalarTensor(s);
  } else {
    AT_ASSERT(s.isIntegral());
    return CPU(kLong).scalarTensor(s);
  }
}

}
