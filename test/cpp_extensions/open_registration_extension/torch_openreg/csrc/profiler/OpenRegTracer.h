#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace torch_openreg::profiler {

// Activity types recorded by the tracer.  The session maps these to
// libkineto::ActivityType when building GenericTraceActivity.
enum class ActivityKind : uint8_t {
  KERNEL,
  MEMCPY,
  MEMSET,
  RUNTIME,
  DRIVER,
};

struct TraceRecord {
  ActivityKind kind;
  std::string name;
  int64_t start; // nanoseconds (c10::getTime)
  int64_t end;
  int32_t device_id;
  int32_t stream_id;
  uint64_t correlation_id;
};

// Singleton tracing engine for the OpenReg backend.
//
// Records device-side events (kernel launches, memory ops, runtime calls)
// while profiling is active. The session (IActivityProfilerSession) calls
// enable()/disable() to control recording, and flush() to harvest records.
//
// Correlation IDs are managed via a thread-local stack driven by PyTorch's
// pushCorrelationId / popCorrelationId around each ATen operator dispatch.
//
// When disabled, isEnabled() is a single atomic load — the only cost on
// the hot path.
class OpenRegTracer {
 public:
  static OpenRegTracer& instance();

  void enable();
  void disable();
  bool isEnabled() const;

  void pushCorrelation(uint64_t id);
  void popCorrelation();
  uint64_t currentCorrelation() const;

  void record(TraceRecord rec);
  std::vector<TraceRecord> flush();

 private:
  OpenRegTracer() = default;

  std::atomic<bool> enabled_{false};

  std::mutex mu_;
  std::vector<TraceRecord> records_;
};

} // namespace torch_openreg::profiler
