#include "torch/csrc/jit/script/logging.h"

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

std::unordered_map<std::string, int64_t> LockingLogger::getCounters() const {
  std::unordered_map<std::string, int64_t> counters;
  std::unique_lock<std::mutex> lk(m);
  for (auto& kv : raw_counters) {
    AggregationType type = agg_types.count(kv.first) ? agg_types.at(kv.first)
                                                     : AggregationType::SUM;
    switch (type) {
      case AggregationType::SUM: {
        float sum = 0;
        for (auto x : kv.second) {
          sum += x;
        }
        counters[kv.first] = sum;
      } break;
      case AggregationType::AVG: {
        float avg = 0;
        for (auto x : kv.second) {
          avg += x;
        }
        avg /= kv.second.size();
        counters[kv.first] = avg;
      } break;
    }
  }
  return counters;
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
  auto seconds = std::chrono::duration<double>(end - tp.point).count();
  logging::getLogger()->addStatValue(name, seconds);
}

} // namespace logging
} // namespace jit
} // namespace torch
