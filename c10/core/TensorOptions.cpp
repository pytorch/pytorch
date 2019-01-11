#include <c10/core/TensorOptions.h>

#include <c10/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include <iostream>

namespace c10 {

#if !C10_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

thread_local bool NonVariableTypeMode_enabled = false;

bool NonVariableTypeMode::is_enabled() {
  return NonVariableTypeMode_enabled;
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  NonVariableTypeMode_enabled = enabled;
}

#else // C10_MOBILE

bool NonVariableTypeMode::is_enabled() {
  throw std::runtime_error("NonVariableTypeMode is not supported on mobile");
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  throw std::runtime_error("NonVariableTypeMode is not supported on mobile");
}

#endif // C10_MOBILE

std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options) {
  return stream << "TensorOptions(dtype=" << options.dtype()
                << ", device=" << options.device()
                << ", layout=" << options.layout()
                << ", requires_grad=" << std::boolalpha
                << options.requires_grad() << ")";
}

} // namespace c10
