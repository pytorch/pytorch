#pragma once

#include <string>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace torch { namespace jit { namespace logging {

class LoggerBase {
 public:
  virtual void addStatValue(std::string stat_name, float val) = 0;
  virtual std::unordered_map<std::string, float> getCounters() const = 0;
  virtual ~LoggerBase() {}
};

std::shared_ptr<LoggerBase> getLogger();
void setLogger(std::shared_ptr<LoggerBase> logger);

// No-op logger. This is the default and is meant to incur almost no runtime
// overhead.

class NoopLogger : public LoggerBase {
 public:
  void addStatValue(std::string stat_name, float val) override {}
  std::unordered_map<std::string, float> getCounters() const override {
    return {};
  }
  ~NoopLogger() {}
};

// Trivial locking logger. Pass in an instance of this to setLogger() to use it.
// This keeps track of the sum of all statistics.
//
// NOTE: this is not written in a scalable way and should probably only be used
// in the single-threaded case or for testing.
class LockingLogger : public LoggerBase {
 public:
  void addStatValue(std::string stat_name, float val) override;
  std::unordered_map<std::string, float> getCounters() const override;
  ~LockingLogger() {}
 private:
  mutable std::mutex m;
  std::unordered_map<std::string, float> counters;
};

}}}