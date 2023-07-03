#include <torch/csrc/jit/runtime/logging.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace logging {

// TODO: multi-scale histogram for this thing

void LockingLogger::addStatValue(const std::string& stat_name, int64_t val) {
  std::unique_lock<std::mutex> lk(m);
  auto& raw_counter = raw_counters[stat_name];
  raw_counter.sum += val;
  raw_counter.count++;
}

int64_t LockingLogger::getCounterValue(const std::string& name) const {
  std::unique_lock<std::mutex> lk(m);
  if (!raw_counters.count(name)) {
    return 0;
  }
  AggregationType type =
      agg_types.count(name) ? agg_types.at(name) : AggregationType::SUM;
  const auto& raw_counter = raw_counters.at(name);
  switch (type) {
    case AggregationType::SUM: {
      return raw_counter.sum;
    } break;
    case AggregationType::AVG: {
      return raw_counter.sum / raw_counter.count;
    } break;
  }
  throw std::runtime_error("Unknown aggregation type!");
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

LoggerBase* setLogger(LoggerBase* logger) {
  LoggerBase* previous = global_logger.load();
  while (!global_logger.compare_exchange_strong(previous, logger)) {
    previous = global_logger.load();
  }
  return previous;
}

JITTimePoint timePoint() {
  return JITTimePoint{std::chrono::high_resolution_clock::now()};
}

void recordDurationSince(const std::string& name, const JITTimePoint& tp) {
  auto end = std::chrono::high_resolution_clock::now();
  // Measurement in microseconds.
  auto seconds = std::chrono::duration<double>(end - tp.point).count() * 1e9;
  logging::getLogger()->addStatValue(name, seconds);
}

} // namespace logging
} // namespace jit
} // namespace torch
