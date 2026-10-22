/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <deque>
#include <memory>
#include <set>

#include "include/IActivityProfiler.h"
#include "output_base.h"

namespace libkineto {

class MockProfilerSession : public IActivityProfilerSession {
 public:
  explicit MockProfilerSession() = default;

  void start() override {
    start_count++;
    status_ = TraceStatus::RECORDING;
  }

  void stop() override {
    stop_count++;
    status_ = TraceStatus::PROCESSING;
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void processTrace(ActivityLogger& logger) override;

  void set_test_activities(std::deque<GenericTraceActivity>&& acs) {
    test_activities_ = std::move(acs);
  }

  std::unique_ptr<CpuTraceBuffer> getTraceBuffer() override;

  std::unique_ptr<DeviceInfo> getDeviceInfo() override {
    return {};
  }

  std::vector<ResourceInfo> getResourceInfos() override {
    return {};
  }

  int start_count = 0;
  int stop_count = 0;

 private:
  std::deque<GenericTraceActivity> test_activities_;
};

class MockActivityProfiler : public IActivityProfiler {
 public:
  explicit MockActivityProfiler(std::deque<GenericTraceActivity>& activities);

  [[nodiscard]] const std::string& name() const override;

  [[nodiscard]] const std::set<ActivityType>& availableActivities()
      const override;

  std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const Config& config) override;

  std::unique_ptr<IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const Config& config) override;

 private:
  std::deque<GenericTraceActivity> test_activities_;
};

} // namespace libkineto
