/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "CuptiPMSamplingController.h"
#include "IActivityProfiler.h"

namespace libkineto {
struct CpuTraceBuffer;
}

namespace KINETO_NAMESPACE {

// TODO: Support dynamic collection toggling
class CuptiPMSamplingSession final
    : public libkineto::IActivityProfilerSession {
 public:
  explicit CuptiPMSamplingSession(const CuptiPMSamplingConfig& config);
  explicit CuptiPMSamplingSession(
      std::unique_ptr<ICuptiPMSamplingController> controller);
  ~CuptiPMSamplingSession() override;

  [[nodiscard]] bool prepare();
  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t startTime,
      int64_t endTime) override;
  [[nodiscard]] std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  [[nodiscard]] std::vector<libkineto::ResourceInfo> getResourceInfos()
      override;
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer()
      override;

 private:
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> buildTraceBuffer();

  std::unique_ptr<ICuptiPMSamplingController> controller_;
  std::unique_ptr<libkineto::CpuTraceBuffer> traceBuffer_;
};

// The naming is unfortunate here, because CUPTI PM sampling is not activity
// profiling. However, Kineto's registration logic is based on
// IActivityProfiler, so this is the simplest way to introduce a new profiler.
class CuptiPMSamplingProfiler final : public libkineto::IActivityProfiler {
 public:
  [[nodiscard]] const std::string& name() const override;
  [[nodiscard]] const std::set<libkineto::ActivityType>& availableActivities()
      const override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  [[nodiscard]] std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t startTimeMs,
      int64_t durationMs,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

} // namespace KINETO_NAMESPACE
