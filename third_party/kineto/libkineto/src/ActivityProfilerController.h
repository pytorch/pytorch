/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityLoggerFactory.h"
#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "AsyncActivityProfilerHandler.h"
#include "ConfigLoader.h"
#include "GenericActivityProfiler.h"
#include "InvariantViolations.h"
#include "LoggerCollector.h"
#include "SyncActivityProfilerHandler.h"

namespace KINETO_NAMESPACE {

class Config;

class ActivityProfilerController : public ConfigLoader::ConfigHandler {
 public:
  explicit ActivityProfilerController(ConfigLoader& configLoader, bool cpuOnly);
  ActivityProfilerController(const ActivityProfilerController&) = delete;
  ActivityProfilerController& operator=(const ActivityProfilerController&) =
      delete;

  ~ActivityProfilerController() override;

#if !USE_GOOGLE_LOG
  static void addLoggerCollectorFactory(
      const std::function<std::shared_ptr<LoggerCollector>()>& factory);
  static std::vector<std::shared_ptr<LoggerCollector>> getLoggerCollectors();
#endif // !USE_GOOGLE_LOG

  static void addLoggerFactory(
      const std::string& protocol,
      ActivityLoggerFactory::FactoryFunc factory);

  static void setInvariantViolationsLoggerFactory(
      const std::function<std::unique_ptr<InvariantViolationsLogger>()>&
          factory);

  // ConfigLoader::ConfigHandler callback API.
  bool canAcceptConfig() override;
  bool acceptConfig(const Config& config) override;

  // These API are used for On-Demand Tracing.
  void asyncScheduleTrace(const Config& config);
  void asyncStep();

  // These API are used for Synchronous Tracing.
  void syncPrepareTrace(const Config& config);
  void syncToggleCollectionDynamic(const bool enable);
  void syncStartTrace();
  std::unique_ptr<ActivityTraceInterface> syncStopTrace();

  bool isActive();
  bool isStopped() const;

  void transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace);

  void recordThreadInfo();

  void addChildActivityProfiler(std::unique_ptr<IActivityProfiler> profiler);

  void addMetadata(const std::string& key, const std::string& value);

  void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id = "");

  void pushCorrelationId(uint64_t id);

  void popCorrelationId();

  void pushUserCorrelationId(uint64_t id);

  void popUserCorrelationId();

  static std::unique_ptr<ActivityLogger> makeLogger(const Config& config);

  static ActivityLoggerFactory& loggerFactory();

 private:
  std::unique_ptr<GenericActivityProfiler> profiler_;
  std::vector<std::shared_ptr<LoggerCollector>> loggerCollectors_;
  ConfigLoader& configLoader_;

  std::unique_ptr<SyncActivityProfilerHandler> syncHandler_;
  std::unique_ptr<AsyncActivityProfilerHandler> asyncHandler_;
};

} // namespace KINETO_NAMESPACE
