/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityProfilerInterface.h"

#include <memory>
#include <set>
#include <vector>

#include "ActivityType.h"
#include "ITraceActivity.h"

namespace libkineto {
// previous declaration is struct so this one must be too.
struct CpuTraceBuffer;
} // namespace libkineto

namespace KINETO_NAMESPACE {

using namespace libkineto;

class ActivityProfilerController;
class Config;
class ConfigLoader;

class ActivityProfilerProxy : public ActivityProfilerInterface {
 public:
  ActivityProfilerProxy(bool cpuOnly, ConfigLoader& configLoader);
  ~ActivityProfilerProxy() override;
  ActivityProfilerProxy(const ActivityProfilerProxy&) = delete;
  ActivityProfilerProxy& operator=(const ActivityProfilerProxy&) = delete;
  ActivityProfilerProxy(ActivityProfilerProxy&&) = delete;
  ActivityProfilerProxy& operator=(ActivityProfilerProxy&&) = delete;

  void init() override;
  bool isInitialized() override {
    return controller_ != nullptr;
  }

  bool isActive() override;
  bool isStopped() const override;

  // These API are used for async/on-demand tracing.
  void scheduleTrace(const std::string& configStr) override;
  void scheduleTrace(const Config& config);
  void step() override;

  // These API are used for sync/auto-trace tracing.
  void prepareTrace(
      const std::set<ActivityType>& activityTypes,
      const std::string& configStr = "") override;

  void toggleCollectionDynamic(const bool enable) override;

  void startTrace() override;
  std::unique_ptr<ActivityTraceInterface> stopTrace() override;

  // TraceActivity API.
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

  void transferCpuTrace(std::unique_ptr<CpuTraceBuffer> traceBuffer) override;

  void recordThreadInfo() override;

  void addMetadata(const std::string& key, const std::string& value) override;

  void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) override;

  void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id = "") override;

 private:
  bool cpuOnly_{true};
  ConfigLoader& configLoader_;
  ActivityProfilerController* controller_{nullptr};
};

} // namespace KINETO_NAMESPACE
