/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "IActivityProfiler.h"

namespace KINETO_NAMESPACE {

std::string getXpuDeviceProperties();

class XPUActivityProfiler : public libkineto::IActivityProfiler {
 public:
  XPUActivityProfiler() = default;
  XPUActivityProfiler(const XPUActivityProfiler&) = delete;
  XPUActivityProfiler& operator=(const XPUActivityProfiler&) = delete;

  const std::string& name() const override {
    return name_;
  }

  [[noreturn]] const std::set<ActivityType>& availableActivities()
      const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const libkineto::Config& config) override;

 private:
  std::string name_{"__xpu_profiler__"};
};

} // namespace KINETO_NAMESPACE
