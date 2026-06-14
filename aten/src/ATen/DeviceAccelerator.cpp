#include <ATen/Context.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <atomic>
#include <cstdint>

namespace at::accelerator {

namespace {

constexpr int16_t kUnsetAccelerator = -1;
std::atomic<int16_t> current_accelerator{kUnsetAccelerator};

bool isAcceleratorDeviceType(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCUDA:
    case at::kMTIA:
    case at::kXPU:
    case at::kHIP:
    case at::kMPS:
    case at::kHPU:
    case at::kPrivateUse1:
    case c10::DeviceType::PrivateUse2:
    case c10::DeviceType::PrivateUse3:
      return true;
    default:
      return false;
  }
}

bool isSelectableAccelerator(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kPrivateUse1:
    case c10::DeviceType::PrivateUse2:
    case c10::DeviceType::PrivateUse3:
      return c10::is_privateuse_backend_registered(device_type);
    case at::kMTIA:
      return at::hasMTIA();
    case at::kCUDA:
      return at::detail::getCUDAHooks().isBuilt();
    case at::kXPU:
      return at::detail::getXPUHooks().isBuilt();
    case at::kHIP:
      return at::detail::getHIPHooks().isBuilt();
    case at::kMPS:
      return at::detail::getMPSHooks().isBuilt();
    case at::kHPU:
      return at::detail::getHPUHooks().isBuilt();
    default:
      return false;
  }
}

std::optional<c10::DeviceType> getCurrentAcceleratorOverride() {
  const auto device_type = current_accelerator.load(std::memory_order_acquire);
  if (device_type == kUnsetAccelerator) {
    return std::nullopt;
  }
  return static_cast<c10::DeviceType>(device_type);
}

} // namespace

std::optional<c10::DeviceType> getAccelerator(bool checked) {
  if (auto device_type = getCurrentAcceleratorOverride()) {
    return device_type;
  }

  // 1. Check runtime backends
  // This state is temporary, these runtime checks should be moved to compile-time
  // once they provide the new isBuilt API and we are sure they're never in the
  // same binary as another accelerator.
#define DETECT_RUNTIME_ACCELERATOR(device_name)     \
  if (at::has##device_name()) {                     \
    return k##device_name;                          \
  }

  DETECT_RUNTIME_ACCELERATOR(MTIA)

#undef DETECT_RUNTIME_ACCELERATOR

  // 2. Check compile-time backends
  std::optional<c10::DeviceType> device_type = std::nullopt;

#define DETECT_AND_ASSIGN_ACCELERATOR_COMP(device_name) \
  if (at::detail::get##device_name##Hooks().isBuilt()) {  \
    TORCH_CHECK(                                         \
        !device_type.has_value(),                        \
        "Cannot have both " #device_name " and ",             \
        device_type.value(), ".");                       \
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
  return isAcceleratorDeviceType(device_type);
}

void setCurrentAccelerator(c10::DeviceType device_type) {
  TORCH_CHECK(
      isAcceleratorDeviceType(device_type),
      c10::DeviceTypeName(device_type),
      " is not an accelerator.");
  TORCH_CHECK(
      isSelectableAccelerator(device_type),
      "Cannot set current accelerator to ",
      c10::DeviceTypeName(device_type),
      " because it is not available in this PyTorch build.");
  current_accelerator.store(
      static_cast<int16_t>(device_type), std::memory_order_release);
}

// NOLINTBEGIN(bugprone-unchecked-optional-access)
c10::DeviceIndex deviceCount() {
  const auto device_type = getAccelerator(false);
  if (!device_type.has_value()) {
    return static_cast<c10::DeviceIndex>(0);
  }
  c10::impl::VirtualGuardImpl impl(device_type.value());
  return impl.deviceCount();
}

void setDeviceIndex(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.setDevice({device_type, device_index});
}

c10::DeviceIndex getDeviceIndex() {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDevice().index();
}

void setCurrentStream(c10::Stream stream) {
  const auto device_type = getAccelerator(true).value();
  TORCH_CHECK(
      device_type == stream.device_type(),
      "stream's device type ",
      c10::DeviceTypeName(stream.device_type()),
      " doesn't match the current accelerator ",
      c10::DeviceTypeName(device_type));
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.exchangeStream(stream);
}

c10::Stream getCurrentStream(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getStream({device_type, device_index});
}

void synchronizeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // impl.synchronizeDevice should can be safely called from any device
  impl.synchronizeDevice(device_index);
}

c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.exchangeDevice({device_type, device_index}).index();
}

c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // Avoid creating a new context if the context for the given device_index
  // is not initialized.
  impl.uncheckedSetDevice({device_type, device_index});
  return impl.getDevice().index();
}

c10::DeviceCapability getDeviceCapability(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDeviceCapability({device_type, device_index});
}

void emptyHostCache() {
  const auto device_type = getAccelerator(true).value();
  at::getHostAllocator(device_type)->empty_cache();
}
// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace at::accelerator
