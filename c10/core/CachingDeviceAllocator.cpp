#include <c10/core/CachingDeviceAllocator.h>

namespace c10::CachingDeviceAllocator {

namespace {

static std::array<CachingDeviceAllocator*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
static std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

} // anonymous namespace

void SetAllocator(
    at::DeviceType t,
    CachingDeviceAllocator* alloc,
    uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

CachingDeviceAllocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

} // namespace c10::CachingDeviceAllocator
