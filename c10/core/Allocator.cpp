#include <c10/core/Allocator.h>

#include <c10/util/ThreadLocalDebugInfo.h>

namespace c10 {

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    const std::function<void(void*)>& deleter,
    Device device) {
  return {ptr,
          new InefficientStdFunctionContext({ptr, deleter}),
          &deleteInefficientStdFunctionContext,
          device};
}

C10_API at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];
C10_API uint8_t allocator_priority[at::COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

void SetAllocator(at::DeviceType t, at::Allocator* alloc, uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  AT_ASSERTM(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

C10_API bool memoryProfilingEnabled() {
  const auto& state = ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE);
  auto reporter_ptr = dynamic_cast<MemoryUsageReporter*>(state.get());
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(Device device, int64_t alloc_size) {
  const auto& state = ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE);
  auto reporter_ptr = dynamic_cast<MemoryUsageReporter*>(state.get());
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(device, alloc_size);
  }
}

} // namespace c10
