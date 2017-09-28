/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  auto v = std::unique_ptr<StatValue>(new StatValue);
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

StatRegistry::~StatRegistry() {}

StatRegistry& StatRegistry::get() {
  static StatRegistry r;
  return r;
}
}
