#include "torch/csrc/jit/script/logging.h"

#include <mutex>
#include <unordered_map>

namespace torch { namespace jit { namespace logging {

// TODO: multi-scale histogram for this thing

void LockingLogger::addStatValue(std::string stat_name, float val) {
  std::unique_lock<std::mutex> lk(m);
  raw_counters[stat_name].push_back(val);
}

std::unordered_map<std::string, float> LockingLogger::getCounters() const {
  std::unordered_map<std::string, float> counters;
  std::unique_lock<std::mutex> lk(m);
  for (auto &kv : raw_counters) {
    AggregationType type = agg_types.count(kv.first) ? agg_types.at(kv.first) : AggregationType::SUM;
    switch (type) {
      case AggregationType::SUM: {
        float sum = 0;
        for (auto x : kv.second) sum += x;
        counters[kv.first] = sum;
      } break;
      case AggregationType::AVG: {
        float avg = 0;
        for (auto x : kv.second) avg += x;
        avg /= kv.second.size();
        counters[kv.first] = avg;
      } break;
    }
  }
  return counters;
}

void LockingLogger::setAggregationType(std::string stat_name, AggregationType type) {
  agg_types[stat_name] = type;
}

// TODO: SLOW
std::mutex m;
std::shared_ptr<LoggerBase> global_logger = std::make_shared<NoopLogger>();

std::shared_ptr<LoggerBase> getLogger() {
  std::unique_lock<std::mutex> lk(m);
  return global_logger;
}

void setLogger(std::shared_ptr<LoggerBase> logger) {
  std::unique_lock<std::mutex> lk(m);
  global_logger = std::move(logger);
}

JITTimePoint timePoint() {
  return JITTimePoint{std::chrono::high_resolution_clock::now()};
}

void recordDurationSince(std::string name, JITTimePoint tp) {
  auto end = std::chrono::high_resolution_clock::now();
  auto seconds = std::chrono::duration<double>(end-tp.point).count();
  logging::getLogger()->addStatValue(name, seconds);
}

}}}
