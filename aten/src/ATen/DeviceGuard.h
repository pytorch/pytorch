#pragma once

#include <c10/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ScalarType.h> // TensorList whyyyyy

namespace at {

inline optional<Device> device_of(Tensor t) {
  if (t.defined()) {
    return make_optional(t.device());
  } else {
    return nullopt;
  }
}

inline optional<Device> device_of(TensorList t) {
  if (!t.empty()) {
    return device_of(t.front());
  } else {
    return nullopt;
  }
}

} // namespace at
