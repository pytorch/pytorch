#include <ATen/Context.h>
#include <ATen/accelerator/Accelerator.h>

namespace at::accelerator {

std::optional<c10::DeviceType> getAccelerator(bool checked) {
  // 1. Check PrivateUse1 backends
  // We explicitly allow PrivateUse1 and another device at the same time as we
  // use this for testing. Whenever a PrivateUse1 device is registered, use it
  // first.
  // Note that this check is only for hook registration and thus is NOT
  // initializing the device or poisoning fork.
  if (is_privateuse1_backend_registered()) {
    return kPrivateUse1;
  }

  // 2. Check runtime backends
  // This state is temporary, these runtime checks should be moved to
  // compile-time once they provide the new isBuilt API and we are sure they're
  // never in the same binary as another accelerator.
#define DETECT_RUNTIME_ACCELERATOR(device_name) \
  if (at::has##device_name()) {                 \
    return k##device_name;                      \
  }

  DETECT_RUNTIME_ACCELERATOR(MTIA)

#undef DETECT_RUNTIME_ACCELERATOR

  // 2. Check compile-time backends
  std::optional<c10::DeviceType> device_type = std::nullopt;

#define DETECT_AND_ASSIGN_ACCELERATOR_COMP(device_name)  \
  if (at::detail::get##device_name##Hooks().isBuilt()) { \
    TORCH_CHECK(                                         \
        !device_type.has_value(),                        \
        "Cannot have both " #device_name " and ",        \
        device_type.value(),                             \
        ".");                                            \
    device_type = k##device_name;                        \
  }

  DETECT_AND_ASSIGN_ACCELERATOR_COMP(CUDA)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(XPU)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(HIP)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(MPS)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(HPU)
  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef DETECT_AND_ASSIGN_ACCELERATOR_COMP
}

bool isAccelerator(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCUDA:
    case at::kMTIA:
    case at::kXPU:
    case at::kHIP:
    case at::kMPS:
    case at::kHPU:
    case at::kPrivateUse1:
      return true;
    default:
      return false;
  }
}
} // namespace at::accelerator
