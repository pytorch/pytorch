#pragma once

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>

// This is in the c10 namespace because we use ADL to find the functions in it.
namespace c10 {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no way
// to implement this without going through Derived Types (which are not part of core).
inline at::Tensor scalar_to_tensor(Scalar s, const Device device = at::kCPU) {
  // This is the fast track we have for CPU scalar tensors.
  if (device == at::kCPU) {
    if (s.isFloatingPoint()) {
      return at::native::scalar_tensor(s, at::device(at::kCPU).dtype(at::kDouble));
    } else if (s.isBoolean()) {
      return at::native::scalar_tensor(s, at::device(at::kCPU).dtype(at::kBool));
    } else if (s.isComplex()) {
      return at::native::scalar_tensor(s, at::device(at::kCPU).dtype(at::kComplexDouble));
    } else {
      AT_ASSERT(s.isIntegral(false));
      return at::native::scalar_tensor(s, at::device(at::kCPU).dtype(at::kLong));
    }
  }
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kDouble));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kComplexDouble));
  } else {
    AT_ASSERT(s.isIntegral(false));
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

}
