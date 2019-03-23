#include "torch/csrc/jit/script/logging.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace torch {
namespace jit {
namespace logging {

// TODO: multi-scale histogram for this thing

void LockingLogger::addStatValue(const std::string& stat_name, int64_t val) {
  std::unique_lock<std::mutex> lk(m);
  raw_counters[stat_name].push_back(val);
}

TORCH_API int64_t LockingLogger::getCounterValue(const std::string& name) const {
  const std::vector<int64_t> *vals;
  int64_t retval;
  AggregationType type;
  std::unique_lock<std::mutex> lk(m);
  if (!raw_counters.count(name)) {
    return 0;
  }
  type = agg_types.count(name) ? agg_types.at(name)
                                              : AggregationType::SUM;
  vals = &raw_counters.at(name);
  switch (type) {
    case AggregationType::SUM: {
      float sum = 0;
      for (auto x : *vals) {
        sum += x;
      }
      retval = sum;
    } break;
    case AggregationType::AVG: {
      float avg = 0;
      for (auto x : *vals) {
        avg += x;
      }
      avg /= vals->size();
      retval = avg;
    } break;
  }
  return retval;
}

void LockingLogger::setAggregationType(
    const std::string& stat_name,
    AggregationType type) {
  agg_types[stat_name] = type;
}


std::atomic<LoggerBase*> global_logger{new NoopLogger()};

LoggerBase* getLogger() {
  return global_logger.load();
}

LoggerBase *setLogger(LoggerBase* logger) {
  LoggerBase *previous = global_logger.load();
  while (!global_logger.compare_exchange_strong(previous, logger)) {
    previous = global_logger.load();
  }
  return previous;
}

JITTimePoint timePoint() {
  return JITTimePoint{std::chrono::high_resolution_clock::now()};
}

void recordDurationSince(const std::string& name, JITTimePoint tp) {
  auto end = std::chrono::high_resolution_clock::now();
  // Measurement in microseconds.
  auto seconds = std::chrono::duration<double>(end - tp.point).count() * 1e6;
  logging::getLogger()->addStatValue(name, seconds);
}

} // namespace logging
} // namespace jit
} // namespace torch
