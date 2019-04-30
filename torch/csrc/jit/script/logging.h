#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace logging {

class LoggerBase {
 public:
  TORCH_API virtual void addStatValue(
      const std::string& stat_name,
      int64_t val) = 0;
  virtual ~LoggerBase() {}
};

TORCH_API LoggerBase* getLogger();
TORCH_API LoggerBase* setLogger(LoggerBase* logger);

// No-op logger. This is the default and is meant to incur almost no runtime
// overhead.

class NoopLogger : public LoggerBase {
 public:
  void addStatValue(const std::string& stat_name, int64_t val) override {}
  ~NoopLogger() {}
};

// Trivial locking logger. Pass in an instance of this to setLogger() to use it.
// This keeps track of the sum of all statistics.
//
// NOTE: this is not written in a scalable way and should probably only be used
// in the single-threaded case or for testing.
class TORCH_API LockingLogger : public LoggerBase {
 public:
  void addStatValue(const std::string& stat_name, int64_t val) override;
  virtual int64_t getCounterValue(const std::string& name) const;
  enum class AggregationType { SUM, AVG };
  void setAggregationType(
      const std::string& stat_name,
      AggregationType type);
  ~LockingLogger() {}

 private:
  mutable std::mutex m;
  struct RawCounter {
    RawCounter() : sum(0), count(0) {}
    int64_t sum;
    size_t count;
  };
  std::unordered_map<std::string, RawCounter> raw_counters;
  std::unordered_map<std::string, AggregationType> agg_types;
};

// Make this struct so the timer internals are opaque to the user.
struct JITTimePoint {
  std::chrono::time_point<std::chrono::high_resolution_clock> point;
};

TORCH_API JITTimePoint timePoint();
TORCH_API void recordDurationSince(const std::string& name, JITTimePoint tp);

namespace runtime_counters {
constexpr const char* GRAPH_EXECUTORS_CONSTRUCTED =
    "pytorch_runtime.graph_executors_constructed";
constexpr const char* GRAPH_EXECUTOR_INVOCATIONS =
    "pytorch_runtime.graph_executor_invocations";
constexpr const char* EXECUTION_PLAN_CACHE_HIT =
    "pytorch_runtime.execution_plan_cache_hit";
constexpr const char* EXECUTION_PLAN_CACHE_MISS =
    "pytorch_runtime.execution_plan_cache_miss";

inline std::vector<const char*> allRuntimeCounters() {
  return {GRAPH_EXECUTORS_CONSTRUCTED,
          GRAPH_EXECUTOR_INVOCATIONS,
          EXECUTION_PLAN_CACHE_HIT,
          EXECUTION_PLAN_CACHE_MISS};
}

} // namespace runtime_counters

} // namespace logging
} // namespace jit
} // namespace torch
