#include <c10/core/Allocator.h>
#include <array>

#include <c10/util/ThreadLocalDebugInfo.h>

#include <atomic>
#include <cstring>

namespace c10 {

namespace {
// While no MemoryReportingInfoBase is alive, none can be installed as the
// PROFILER_STATE debug info, so the allocator hooks skip the TLS lookup.
// Relaxed suffices: a thread that sees an installed reporter is already
// ordered after its construction and before its destruction.
std::atomic<size_t> live_memory_reporters{0};

MemoryReportingInfoBase* current_memory_reporter() {
  if (C10_LIKELY(live_memory_reporters.load(std::memory_order_relaxed) == 0)) {
    return nullptr;
  }
  return static_cast<MemoryReportingInfoBase*>(
      ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
}
} // namespace

MemoryReportingInfoBase::MemoryReportingInfoBase() noexcept {
  live_memory_reporters.fetch_add(1, std::memory_order_relaxed);
}

MemoryReportingInfoBase::MemoryReportingInfoBase(
    const MemoryReportingInfoBase& /*other*/) noexcept
    : MemoryReportingInfoBase() {}

MemoryReportingInfoBase::MemoryReportingInfoBase(
    MemoryReportingInfoBase&& /*other*/) noexcept
    : MemoryReportingInfoBase() {}

MemoryReportingInfoBase::~MemoryReportingInfoBase() {
  live_memory_reporters.fetch_sub(1, std::memory_order_relaxed);
}

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

static std::array<at::Allocator*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
static std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

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
  auto* reporter_ptr = current_memory_reporter();
  return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}

void reportMemoryUsageToProfiler(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    Device device) {
  auto* reporter_ptr = current_memory_reporter();
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
  auto* reporter_ptr = current_memory_reporter();
  if (reporter_ptr) {
    reporter_ptr->reportOutOfMemory(
        alloc_size, total_allocated, total_reserved, device);
  }
}

void MemoryReportingInfoBase::reportOutOfMemory(
    int64_t /*alloc_size*/,
    size_t /*total_allocated*/,
    size_t /*total_reserved*/,
    Device /*device*/) {}

} // namespace c10
