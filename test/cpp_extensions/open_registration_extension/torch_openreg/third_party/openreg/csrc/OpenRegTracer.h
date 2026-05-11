#pragma once

#include <atomic>
#include <cstdint>

#include <include/openreg.h>

namespace openreg::profiler {

// Singleton tracer for the OpenReg backend.
//
// Manages the correlation-ID stack driven by PyTorch's
// pushCorrelationId / popCorrelationId around each ATen dispatch.
// The profiler session (IActivityProfilerSession) uses enable()/disable()
// to control recording and generates synthetic device activities on flush.
//
// When disabled, isEnabled() is a single atomic load — the only cost on
// the hot path.
class OPENREG_EXPORT OpenRegTracer {
 public:
  static OpenRegTracer& instance();

  void enable();
  void disable();
  bool isEnabled() const;

  void pushCorrelation(uint64_t id);
  void popCorrelation();
  uint64_t currentCorrelation() const;

 private:
  OpenRegTracer() = default;

  std::atomic<bool> enabled_{false};
};

} // namespace openreg::profiler
