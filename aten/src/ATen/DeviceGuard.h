#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h> // TensorList whyyyyy

namespace at {

// Are you here because you're wondering why DeviceGuard(tensor) no
// longer works?  For code organization reasons, we have temporarily(?)
// removed this constructor from DeviceGuard.  The new way to
// spell it is:
//
//    OptionalDeviceGuard guard(device_of(tensor));

/// Return the Device of a Tensor, if the Tensor is defined.
inline c10::optional<Device> device_of(const Tensor& t) {
  if (t.defined()) {
    return c10::make_optional(t.device());
  } else {
    return c10::nullopt;
  }
}

inline c10::optional<Device> device_of(const c10::optional<Tensor>& t) {
  return t.has_value() ? device_of(t.value()) : c10::nullopt;
}

/// Return the Device of a TensorList, if the list is non-empty and
/// the first Tensor is defined.  (This function implicitly assumes
/// that all tensors in the list have the same device.)
inline c10::optional<Device> device_of(ITensorListRef t) {
  if (!t.empty()) {
    return device_of(t.front());
  } else {
    return c10::nullopt;
  }
}

} // namespace at
