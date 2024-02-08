#include <ATen/DeviceAccelerator.h>
#include <ATen/Context.h>

namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#define CHECK_NO_CUDA \
  TORCH_CHECK(!at::hasCUDA(), "Cannot have both CUDA and PrivateUse1");

#define CHECK_NO_PU1 \
  TORCH_CHECK(!is_privateuse1_backend_registered(), "Cannot have both CUDA and PrivateUse1");

    if (at::hasCUDA()) {
        CHECK_NO_PU1
        return kCUDA;
    } else if (is_privateuse1_backend_registered()) {
        CHECK_NO_CUDA
        return kPrivateUse1;
    } else {
        TORCH_CHECK(!checked, "Cannot access accelerator device when none is available.")
        return std::nullopt;
    }

#undef CHECK_NO_CUDA
#undef CHECK_NO_PU1
}


} // namespace at
