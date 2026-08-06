#include <c10/core/Allocator.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>

#include <c10/util/ThreadLocalDebugInfo.h>

#include <cstring>

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

static std::array<at::Allocator*, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_array{};
static std::array<uint8_t, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    allocator_priority{};

namespace {
std::atomic<bool> global_memory_reporting_enabled{false};
std::mutex global_memory_reporter_mutex;
std::condition_variable global_memory_reporter_cv;
std::shared_ptr<MemoryReportingInfoBase> global_memory_reporter;
size_t global_memory_reports_in_flight{0};

class GlobalMemoryReporterGuard {
 public:
  GlobalMemoryReporterGuard() {
    if (!global_memory_reporting_enabled.load(std::memory_order_acquire)) {
      return;
    }

    std::lock_guard<std::mutex> guard(global_memory_reporter_mutex);
    if (global_memory_reporter) {
      reporter_ = global_memory_reporter;
      ++global_memory_reports_in_flight;
    }
  }

  GlobalMemoryReporterGuard(const GlobalMemoryReporterGuard&) = delete;
  GlobalMemoryReporterGuard& operator=(const GlobalMemoryReporterGuard&) =
      delete;

  ~GlobalMemoryReporterGuard() {
    if (!reporter_) {
      return;
    }

    std::lock_guard<std::mutex> guard(global_memory_reporter_mutex);
    if (--global_memory_reports_in_flight == 0) {
      global_memory_reporter_cv.notify_all();
    }
  }

  MemoryReportingInfoBase* operator->() const {
    return reporter_.get();
  }

  explicit operator bool() const {
    return reporter_ != nullptr;
  }

 private:
  std::shared_ptr<MemoryReportingInfoBase> reporter_;
};
} // namespace

void setGlobalMemoryReportingInfo(
    std::shared_ptr<MemoryReportingInfoBase> reporter) {
  global_memory_reporting_enabled.store(false, std::memory_order_release);
  std::unique_lock<std::mutex> lock(global_memory_reporter_mutex);
  global_memory_reporter.reset();
  global_memory_reporter_cv.wait(
      lock, [] { return global_memory_reports_in_flight == 0; });
  global_memory_reporter = std::move(reporter);
  global_memory_reporting_enabled.store(
      global_memory_reporter != nullptr, std::memory_order_release);
}

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
  if (reporter_ptr) {
    return reporter_ptr->memoryProfilingEnabled();
  }
  GlobalMemoryReporterGuard global_reporter;
  return global_reporter && global_reporter->memoryProfilingEnabled();
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
  } else {
    GlobalMemoryReporterGuard global_reporter;
    if (global_reporter) {
      global_reporter->reportMemoryUsage(
          ptr, alloc_size, total_allocated, total_reserved, device);
    }
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
  } else {
    GlobalMemoryReporterGuard global_reporter;
    if (global_reporter) {
      global_reporter->reportOutOfMemory(
          alloc_size, total_allocated, total_reserved, device);
    }
  }
}

void MemoryReportingInfoBase::reportOutOfMemory(
    int64_t /*alloc_size*/,
    size_t /*total_allocated*/,
    size_t /*total_reserved*/,
    Device /*device*/) {}

} // namespace c10
