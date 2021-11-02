#include "caffe2/core/stats.h"

#include <condition_variable>
#include <thread>

namespace caffe2 {

ExportedStatMap toMap(const ExportedStatList& stats) {
  ExportedStatMap statMap;
  for (const auto& stat : stats) {
    // allow for multiple instances of a key
    statMap[stat.key] += stat.value;
  }
  return statMap;
}

StatValue* StatRegistry::add(const std::string& name) {
  std::lock_guard<std::mutex> lg(mutex_);
  auto it = stats_.find(name);
  if (it != stats_.end()) {
    return it->second.get();
  }
  auto v = std::make_unique<StatValue>();
  auto value = v.get();
  stats_.insert(std::make_pair(name, std::move(v)));
  return value;
}

void StatRegistry::publish(ExportedStatList& exported, bool reset) {
  std::lock_guard<std::mutex> lg(mutex_);
  exported.resize(stats_.size());
  int i = 0;
  for (const auto& kv : stats_) {
    auto& out = exported.at(i++);
    out.key = kv.first;
    out.value = reset ? kv.second->reset() : kv.second->get();
    out.ts = std::chrono::high_resolution_clock::now();
  }
}

void StatRegistry::update(const ExportedStatList& data) {
  for (const auto& stat : data) {
    add(stat.key)->increment(stat.value);
  }
}

// NOLINTNEXTLINE(modernize-use-equals-default)
StatRegistry::~StatRegistry() {}

StatRegistry& StatRegistry::get() {
  static StatRegistry r;
  return r;
}
}
