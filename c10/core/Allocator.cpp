#include <c10/core/Allocator.h>

#include <c10/util/ThreadLocalDebugInfo.h>

namespace c10 {

DataPtr Allocator::clone(const void* data, std::size_t n) {
  DataPtr new_data = allocate(n);
  copy_data(new_data.mutable_get(), data, n);
  return new_data;
}

void Allocator::default_copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  std::memcpy(dest, src, count);
}

bool Allocator::is_simple_data_ptr(const DataPtr& data_ptr) const {
  return data_ptr.get() == data_ptr.get_context();
}

static void deleteInefficientStdFunctionContext(void* ptr) {
  delete static_cast<InefficientStdFunctionContext*>(ptr);
}

at::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    std::function<void(void*)> deleter,
    Device device) {
  return {
      ptr,
      new InefficientStdFunctionContext(ptr, std::move(deleter)),
      &deleteInefficientStdFunctionContext,
      device};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
C10_API at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
C10_API uint8_t allocator_priority[at::COMPILE_TIME_MAX_DEVICE_TYPES] = {0};

void SetAllocator(at::DeviceType t, at::Allocator* alloc, uint8_t priority) {
  if (priority >= allocator_priority[static_cast<int>(t)]) {
    allocator_array[static_cast<int>(t)] = alloc;
    allocator_priority[static_cast<int>(t)] = priority;
  }
}

at::Allocator* GetAllocator(const at::DeviceType& t) {
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}

bool memoryProfilingEnabled() {
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportMemoryUsage(
        ptr, alloc_size, total_allocated, total_reserved, device);
  }
}

void reportOutOfMemoryToProfiler(
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
  if (reporter_ptr) {
    reporter_ptr->reportOutOfMemory(
        alloc_size, total_allocated, total_reserved, device);
  }
}

MemoryReportingInfoBase::MemoryReportingInfoBase() = default;

void MemoryReportingInfoBase::reportOutOfMemory(
    int64_t /*alloc_size*/,
    size_t /*total_allocated*/,
    size_t /*total_reserved*/,
    Device /*device*/) {}

} // namespace c10
