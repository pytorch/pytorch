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

// The above function is useful for type promotion
// in Binary Ops where one argument is `Tensor` and other is `Scalar`.
// In the above function, we generate wrapped tensor to type with highest
// range and precision based on scalar's type (to support type promotion).
// Eg. Floating Point Types -> Double
//     Complex Types -> Complex Double
//
// However for `Scalar-Scalar` Binary Op,we default the type of wrapped tensor
// to the default type corresponding to scalar's type.
inline at::Tensor scalar_to_tensor_default_dtype(
    Scalar s,
    const Device device = at::kCPU) {
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_dtype()));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(
        s, at::device(device).dtype(at::get_default_complex_dtype()));
  } else {
    AT_ASSERT(s.isIntegral(false));
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

}
