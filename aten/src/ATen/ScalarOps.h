#pragma once

#include <c10/core/Scalar.h>
#include "ATen/Tensor.h"

namespace c10 {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no way
// to implement this without going through Derived Types (which are not part of core).
inline at::Tensor scalar_to_tensor(Scalar s) {
  if (s.isFloatingPoint()) {
    return at::CPU(kDouble).scalarTensor(s);
  } else {
    AT_ASSERT(s.isIntegral());
    return at::CPU(kLong).scalarTensor(s);
  }
}

}
