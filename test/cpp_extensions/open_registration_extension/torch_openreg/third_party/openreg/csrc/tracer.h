#pragma once

#include <atomic>
#include <cstdint>

#include <include/openreg.h>

namespace openreg::profiler {

// Singleton tracer for the OpenReg backend — mirrors CUPTI activity API
// semantics (cuptiActivityPushExternalCorrelationId, etc.).
//
// PyTorch calls pushExternalCorrelationId / popExternalCorrelationId around
// each ATen dispatch. The profiler session uses enableActivityTracing() /
// disableActivityTracing() to control recording.
//
// When disabled, isActivityTracingEnabled() is a single atomic load — the
// only cost on the hot path.
class OPENREG_EXPORT OpenRegTracer {
 public:
  static OpenRegTracer& instance();

  void enableActivityTracing();
  void disableActivityTracing();
  bool isActivityTracingEnabled() const;

  void pushExternalCorrelationId(uint64_t id);
  void popExternalCorrelationId();
  uint64_t getExternalCorrelationId() const;

 private:
  OpenRegTracer() = default;

  std::atomic<bool> enabled_{false};
};

} // namespace openreg::profiler
