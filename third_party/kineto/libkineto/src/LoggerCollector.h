/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !USE_GOOGLE_LOG

#include <atomic>
#include <map>
#include <set>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ILoggerObserver.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    // Skip STAGE output type which is only used by USTLoggerCollector.
    // Skip VERBOSE output type which may bloat metadata section of traces.
    if (ot == STAGE || ot == VERBOSE) {
      return;
    }
    buckets_[ot].push_back(message);
  }

  const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() override {
    return buckets_;
  }

  void reset() override {
    buckets_.clear();
    trace_duration_ms = 0;
    event_count = 0;
    destinations.clear();
  }

  void addDevice(const int64_t device) override {
    devices.insert(device);
  }

  void setTraceDurationMS(const int64_t duration) override {
    trace_duration_ms = duration;
  }

  void addEventCount(const int64_t count) override {
    event_count += count;
  }

  void addDestination(const std::string& dest) override {
    if (!dest.empty()) {
      destinations.insert(dest);
    }
  }

  void addMetadata(
      [[maybe_unused]] const std::string& key,
      [[maybe_unused]] const std::string& value) override {}

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

  // These are useful metadata to collect from CUPTIActivityProfiler for
  // internal tracking.
  std::set<int64_t> devices;
  int64_t trace_duration_ms{0};
  std::atomic<uint64_t> event_count{0};
  std::set<std::string> destinations;
};

} // namespace KINETO_NAMESPACE

#endif // !USE_GOOGLE_LOG
