/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/output_base.h"
#include "src/ActivityBuffers.h"

#include <functional>

namespace KN = KINETO_NAMESPACE;

bool IsEnvVerbose();

// Discards everything: tests inspect the session's trace buffer instead.
class TestActivityLogger : public KN::ActivityLogger {
 public:
  void handleDeviceInfo(
      [[maybe_unused]] const KN::DeviceInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleResourceInfo(
      [[maybe_unused]] const KN::ResourceInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleOverheadInfo(
      [[maybe_unused]] const KN::ActivityLogger::OverheadInfo& info,
      [[maybe_unused]] int64_t time) override {}
  void handleTraceSpan([[maybe_unused]] const KN::TraceSpan& span) override {}
  void handleActivity(
      [[maybe_unused]] const KN::ITraceActivity& activity) override {}
  void handleGenericActivity(
      [[maybe_unused]] const KN::GenericTraceActivity& activity) override {}
  void handleTraceStart(
      [[maybe_unused]] const std::unordered_map<std::string, std::string>&
          metadata,
      [[maybe_unused]] const std::string& device_properties) override {}
  void finalizeMemoryTrace(
      [[maybe_unused]] const std::string&,
      [[maybe_unused]] const KN::Config&) override {}
  void finalizeTrace(
      [[maybe_unused]] const KN::Config& config,
      [[maybe_unused]] std::unique_ptr<KN::ActivityBuffers> buffers,
      [[maybe_unused]] int64_t endTime) override {}
};

std::pair<
    std::unique_ptr<KN::IActivityProfilerSession>,
    std::unique_ptr<KN::CpuTraceBuffer>>
RunProfilerTest(
    const std::vector<std::string_view>& metrics,
    const std::set<KN::ActivityType>& activities,
    const KN::Config& cfg,
    unsigned repeatCount,
    std::vector<std::string_view>&& expectedActivities,
    std::vector<std::string_view>&& expectedTypes,
    int64_t userCorrelationId = 0,
    const KN::ITraceActivity* linkedCpuActivity = nullptr,
    std::function<const KN::ITraceActivity*(int32_t)> linkedActivityCallback =
        nullptr);
