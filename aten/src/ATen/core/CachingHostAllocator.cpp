#include <ATen/core/CachingHostAllocator.h>

#include <array>

namespace at {

namespace {

std::array<HostAllocator*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

} // anonymous namespace

void setHostAllocator(
    at::DeviceType device_type,
    at::HostAllocator* allocator,
    uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(device_type)]) {
    allocator_array[static_cast<int>(device_type)] = allocator;
    allocator_priority[static_cast<int>(device_type)] = priority;
  }
}

at::HostAllocator* getHostAllocator(at::DeviceType device_type) {
  auto* allocator = allocator_array[static_cast<int>(device_type)];
  TORCH_CHECK(
      allocator,
      "Host Allocator for ",
      device_type,
      " is not set, the backend must register one via REGISTER_HOST_ALLOCATOR.");
  return allocator;
}

} // namespace at
