#pragma once

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>

// This is in the c10 namespace because we use ADL to find the functions in it.
namespace c10 {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no way
// to implement this without going through Derived Types (which are not part of core).
inline at::Tensor scalar_to_tensor(Scalar s, const Device device = at::kCPU) {
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kDouble));
  } else {
    AT_ASSERT(s.isIntegral());
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

}
