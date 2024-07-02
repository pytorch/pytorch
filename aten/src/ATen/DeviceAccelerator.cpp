#include <ATen/DeviceAccelerator.h>
#include <ATen/Context.h>

namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#define CHECK_NO_DEVICE_FOR(TYPE) \
  TORCH_CHECK(!at::has##TYPE(), "Cannot have ", #TYPE, " with other devices");

#define CHECK_NO_DEVICE_EXCEPT_CUDA \
  CHECK_NO_DEVICE_FOR(MTIA)         \
  CHECK_NO_DEVICE_FOR(HIP)          \
  CHECK_NO_DEVICE_FOR(MPS)          \
  CHECK_NO_DEVICE_FOR(XPU)

#define CHECK_NO_DEVICE_EXCEPT_MTIA \
  CHECK_NO_DEVICE_FOR(HIP)          \
  CHECK_NO_DEVICE_FOR(MPS)          \
  CHECK_NO_DEVICE_FOR(XPU)

#define CHECK_NO_DEVICE_EXCEPT_HIP  \
  CHECK_NO_DEVICE_FOR(MPS)          \
  CHECK_NO_DEVICE_FOR(XPU)

#define CHECK_NO_DEVICE_EXCEPT_MPS  \
  CHECK_NO_DEVICE_FOR(XPU)

    if (is_privateuse1_backend_registered()) {
        // We explicitly allow PrivateUse1 and another device at the same time
        // as we use this for testing.
        // Whenever a PrivateUse1 device is registered, use it first.
        return kPrivateUse1;
    } else if (at::hasCUDA()) {
        CHECK_NO_DEVICE_EXCEPT_CUDA
        return kCUDA;
    } else if (at::hasMTIA()) {
        CHECK_NO_DEVICE_EXCEPT_MTIA
        return kMTIA;
    } else if (at::hasHIP()) {
        CHECK_NO_DEVICE_EXCEPT_HIP
        return kHIP;
    } else if (at::hasMPS()) {
        CHECK_NO_DEVICE_EXCEPT_MPS
        return kMPS;
    } else if (at::hasXPU()) {
        return kXPU;
    } else {
        TORCH_CHECK(!checked, "Cannot access accelerator device when none is available.")
        return std::nullopt;
    }

#undef CHECK_NO_DEVICE_FOR
#undef CHECK_NO_DEVICE_EXCEPT_CUDA
#undef CHECK_NO_DEVICE_EXCEPT_MTIA
#undef CHECK_NO_DEVICE_EXCEPT_HIP
#undef CHECK_NO_DEVICE_EXCEPT_MPS
}


} // namespace at
