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

void SetAllocator(at::DeviceType t, at::Allocator* alloc) {
  allocator_array[static_cast<int>(t)] = alloc;
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  AT_ASSERTM(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

bool memoryProfilingEnabled() {
  const auto& state = ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE);
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(state.get());
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(void* ptr, int64_t alloc_size, Device device) {
  const auto& state = ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE);
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(state.get());
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(ptr, alloc_size, device);
  }
}

MemoryReportingInfoBase::MemoryReportingInfoBase() {}

} // namespace c10
