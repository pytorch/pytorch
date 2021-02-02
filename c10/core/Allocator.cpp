#include <c10/core/Allocator.h>

#include <c10/util/ThreadLocalDebugInfo.h>
#include <atomic>

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
  TORCH_INTERNAL_ASSERT(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

namespace {
std::atomic<bool> global_memory_reporting_ {false};
}
void enableGlobalMemoryReporting(bool enable) {
  global_memory_reporting_ = enable;
}

bool memoryProfilingEnabled() {
  if (global_memory_reporting_) {
    return true;
  }
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(void* ptr, int64_t alloc_size, Device device) {
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(ptr, alloc_size, device);
  }
}

MemoryReportingInfoBase::MemoryReportingInfoBase() {}

} // namespace c10
