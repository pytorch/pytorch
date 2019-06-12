#pragma once

#include <c10/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ScalarType.h> // TensorList whyyyyy

namespace at {

// Are you here because you're wondering why DeviceGuard(tensor) no
// longer works?  For code organization reasons, we have temporarily(?)
// removed this constructor from DeviceGuard.  The new way to
// spell it is:
//
//    OptionalDeviceGuard guard(device_of(tensor));

/// Return the Device of a Tensor, if the Tensor is defined.
inline optional<Device> device_of(Tensor t) {
  if (t.defined()) {
    return make_optional(t.device());
  } else {
    return nullopt;
  }
}

/// Return the Device of a TensorList, if the list is non-empty and
/// the first Tensor is defined.  (This function implicitly assumes
/// that all tensors in the list have the same device.)
inline optional<Device> device_of(TensorList t) {
  if (!t.empty()) {
    return device_of(t.front());
  } else {
    return nullopt;
  }
}

} // namespace at
