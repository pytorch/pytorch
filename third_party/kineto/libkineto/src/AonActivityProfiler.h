/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IActivityProfiler.h"

namespace KINETO_NAMESPACE {

class IAonTraceConsumer {
 public:
  virtual ~IAonTraceConsumer() = default;
  [[nodiscard]] virtual bool initialize(
      const std::filesystem::path& databasePath) = 0;
  [[nodiscard]] virtual bool start() = 0;
  [[nodiscard]] virtual bool stop() = 0;
};

class AonActivitySession final
    : public libkineto::IActivityProfilerSession {
 public:
  AonActivitySession();
  AonActivitySession(
      std::unique_ptr<IAonTraceConsumer> consumer,
      std::filesystem::path databasePath,
      bool cleanupDatabase = true);
  ~AonActivitySession() override;

  [[nodiscard]] bool prepare();
  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  [[nodiscard]] std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  [[nodiscard]] std::vector<libkineto::ResourceInfo> getResourceInfos()
      override;
  [[nodiscard]] std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer()
      override;

 private:
  void recordError(std::string error);

  std::unique_ptr<IAonTraceConsumer> consumer_;
  std::filesystem::path databasePath_;
  bool cleanupDatabase_{true};
  bool prepared_{false};
  bool started_{false};
  std::vector<std::string> errors_;
};

// Native child-profiler seam for NVIDIA Nsight Perf (AON). Activities remain
// empty until the NvTrace data-model converter is wired in; this prevents an
// intermediate build from claiming that it can produce records it drops.
class AonActivityProfiler final : public libkineto::IActivityProfiler {
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

// The default is CUPTI. AON must be selected explicitly with
// KINETO_GPU_PROFILER=aon.
[[nodiscard]] bool isAonGpuProfilerSelected();

} // namespace KINETO_NAMESPACE
