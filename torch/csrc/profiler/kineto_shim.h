#pragma once

#include <memory>
#include <string>

// Skip Kineto dependency on mobile unless explicitly asked for.
// When is it explicitly asked for?
//   KinetoEdgeCPUProfiler uses KinetoProfiler for cpu
//   event profiling. This has a dependency on cpu only libkineto
#if defined(USE_KINETO) && defined(C10_MOBILE) && \
    !defined(EDGE_PROFILER_USE_KINETO)
#undef USE_KINETO
#endif

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>

#ifdef USE_KINETO
// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto {
enum class ActivityType;
struct CpuTraceBuffer;
class ActivityTraceInterface;
}
#endif

namespace torch {
namespace profiler {

#ifdef USE_KINETO
constexpr bool kKinetoAvailable {true};
#else
constexpr bool kKinetoAvailable {false};
#endif

namespace impl {
namespace kineto {

// ----------------------------------------------------------------------------
// -- Interface (Does not require Kineto) -------------------------------------
// ----------------------------------------------------------------------------
struct DeviceAndResource {
#ifdef USE_KINETO
  int32_t device;
  int32_t resource;
#endif // USE_KINETO
};
const DeviceAndResource kineto_ids();

#ifdef USE_KINETO
using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;
#else
struct DummyTraceBuffer {};
struct DummyTraceInterface {};

using trace_t = DummyTraceBuffer;
using interface_trace_t = DummyTraceBuffer;
#endif // USE_KINETO

// Wraps: libkineto::CpuTraceBuffer
struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);
  TraceWrapper(TraceWrapper&&) = default;
  TraceWrapper(const TraceWrapper&) = delete;

  // The caller is expected to hold a mutex when calling `addCPUActivity` and
  // addMemoryUsageActivity.
  void addCPUActivity(
      const std::string& name,
      const DeviceAndResource device_and_resource,
      const uint64_t correlation_id,
      const int64_t start_time,
      const int64_t end_time);

  void addMemoryUsageActivity(
      const std::string& name,
      const DeviceAndResource device_and_resource,
      const int64_t time,
      const c10::Device device,
      const void* ptr,
      const int64_t alloc_size,
      const int64_t total_allocated,
      const int64_t total_reserved);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() {
      return cpu_trace_;
  }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

// Wraps libkineto::ActivityTraceInterface
struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace);
  ActivityTraceWrapper() = default;
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() {
      return trace_;
  }

 private:
  std::unique_ptr<interface_trace_t> trace_;
  bool saved_ = false; // Kineto's save is destructive
};

using ActivitySet = std::set<torch::autograd::profiler::ActivityType>;
void prepareTrace(const bool cpuOnly, const ActivitySet& activities);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void recordThreadInfo();

} // namespace kineto
} // namespace impl
} // namespace profiler

namespace autograd {
namespace profiler {
#ifdef USE_KINETO
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type);
#endif // USE_KINETO

TORCH_API void addMetadataJson(
    const std::string& key,
    const std::string& value);

} // namespace profiler
} // namespace autograd
} // namespace torch
