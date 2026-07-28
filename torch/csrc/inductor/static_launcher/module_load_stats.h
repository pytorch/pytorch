#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// Header-only telemetry counters for AOTInductor module-load observability.
//
// All counters are DSO-local: each inline function's function-local static has
// one instance per DSO that links it (vague linkage does not unify statics
// across separately-linked DSOs). Read a counter from the same DSO that
// increments it, otherwise the read resolves to that DSO's zero.
//
//  - aotiEngineLoadCounter: AOTInductor engines materialized when a lowered
//    module is loaded. Increment and read from the same TU so the read is
//    same-DSO by construction.
//  - moduleLoadCounter: cu/hipModuleLoad calls issued by the inductor static
//    launcher (static_launcher/cuda.cpp), which compiles into the torch._C
//    Python extension. Reads from a DSO that does not link cuda.cpp resolve to
//    zero -- use aotiEngineLoadCounter there instead.
//  - dtlBytesCounter: bytes copied by the deferred tensor loader (increment and
//    read live in the same TU), for weight-traffic accounting.
//
// Header-only inline so consumers need no link-time dep on the USE_CUDA-gated
// cuda.cpp translation unit; purely additive.

namespace facebook::aoti {

// AOTInductor engine loads (incremented when a lowered module is loaded).
inline std::atomic<int64_t>& aotiEngineLoadCounter() {
  // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
  static std::atomic<int64_t> count{0};
  return count;
}

inline void addAotiEngineLoads(int64_t n) {
  if (n > 0) {
    aotiEngineLoadCounter().fetch_add(n, std::memory_order_relaxed);
  }
}

inline int64_t getCumulativeAotiEngineLoadCount() {
  return aotiEngineLoadCounter().load(std::memory_order_relaxed);
}

// cu/hipModuleLoad count (incremented in static_launcher/cuda.cpp). Reads from
// a DSO that does not link cuda.cpp resolve to zero -- use
// aotiEngineLoadCounter there instead.
inline std::atomic<int64_t>& moduleLoadCounter() {
  // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
  static std::atomic<int64_t> count{0};
  return count;
}

inline int64_t getCumulativeModuleLoadCount() {
  return moduleLoadCounter().load(std::memory_order_relaxed);
}

// Seconds since the FIRST observed load in this DSO. The static initializes on
// first call, not at process start, so this is elapsed-since-first-load, not
// process uptime -- named accordingly to avoid the misread. Consequently the
// log emitted on that first observed load reports 0; the value grows
// monotonically on subsequent loads.
inline int64_t secondsSinceFirstObservedLoad() {
  static const auto first = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now() - first)
      .count();
}

// Cumulative bytes loaded via the deferred tensor loader's per-tensor copy
// path (increment and read live in the same TU).
inline std::atomic<int64_t>& dtlBytesCounter() {
  // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
  static std::atomic<int64_t> bytes{0};
  return bytes;
}

inline int64_t getCumulativeDtlBytes() {
  return dtlBytesCounter().load(std::memory_order_relaxed);
}

inline void addDtlBytes(int64_t n) {
  if (n > 0) {
    dtlBytesCounter().fetch_add(n, std::memory_order_relaxed);
  }
}

} // namespace facebook::aoti
