/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <set>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "libkineto.h"
#include "output_base.h"
#include "test/MockActivitySubProfiler.h"

namespace libkineto {

const std::set<ActivityType> supported_activities{ActivityType::CPU_OP};
const std::string profile_name{"MockProfiler"};

void MockProfilerSession::processTrace(ActivityLogger& logger) {
  for (const auto& activity : test_activities_) {
    activity.log(logger);
  }
}

const std::string& MockActivityProfiler::name() const {
  return profile_name;
}

const std::set<ActivityType>& MockActivityProfiler::availableActivities()
    const {
  return supported_activities;
}

MockActivityProfiler::MockActivityProfiler(
    std::deque<GenericTraceActivity>& activities)
    : test_activities_(activities) {}

std::unique_ptr<IActivityProfilerSession> MockActivityProfiler::configure(
    [[maybe_unused]] const std::set<ActivityType>& activity_types,
    [[maybe_unused]] const Config& config) {
  auto session = std::make_unique<MockProfilerSession>();
  session->set_test_activities(std::move(test_activities_));
  return session;
}

std::unique_ptr<IActivityProfilerSession> MockActivityProfiler::configure(
    [[maybe_unused]] int64_t ts_ms,
    [[maybe_unused]] int64_t duration_ms,
    const std::set<ActivityType>& activity_types,
    const Config& config) {
  return configure(activity_types, config);
}

std::unique_ptr<CpuTraceBuffer> MockProfilerSession::getTraceBuffer() {
  auto buf = std::make_unique<CpuTraceBuffer>();
  for (auto& i : test_activities_) {
    buf->emplace_activity(std::move(i));
  }
  test_activities_.clear();
  return buf;
}
} // namespace libkineto
