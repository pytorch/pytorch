#include <ATen/DeviceAccelerator.h>
#include <ATen/Context.h>

namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#define CHECK_NO_CUDA \
  TORCH_CHECK(!at::hasCUDA(), "Cannot have both CUDA and PrivateUse1");

#define CHECK_NO_PU1 \
  TORCH_CHECK(!is_privateuse1_backend_registered(), "Cannot have both CUDA and PrivateUse1");

    if (is_privateuse1_backend_registered()) {
        // We explicitly allow PrivateUse1 and another device at the same time
        // as we use this for testing.
        // Whenever a PrivateUse1 device is registered, use it first.
        return kPrivateUse1;
    } else if (at::hasCUDA()) {
        CHECK_NO_PU1
        return kCUDA;
    } else {
        TORCH_CHECK(!checked, "Cannot access accelerator device when none is available.")
        return std::nullopt;
    }

#undef CHECK_NO_CUDA
#undef CHECK_NO_PU1
}


} // namespace at
