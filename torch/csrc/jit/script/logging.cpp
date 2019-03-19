#include "torch/csrc/jit/script/logging.h"

#include <mutex>
#include <unordered_map>

namespace torch { namespace jit { namespace logging {

void LockingLogger::addStatValue(std::string stat_name, float val) {
  std::unique_lock<std::mutex> lk(m);
  counters[stat_name] += val;
}

std::unordered_map<std::string, float> LockingLogger::getCounters() const {
  std::unique_lock<std::mutex> lk(m);
  return counters;
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

}}}