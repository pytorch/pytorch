/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Config.h"

#include <chrono>
#include <vector>

namespace KINETO_NAMESPACE {

constexpr char kXpuptiProfilerConfigName[] = "xpupti_scope_profiler";

class XpuptiScopeProfilerConfig : public AbstractConfig {
 public:
  bool handleOption(const std::string& name, std::string& val) override;

  void validate(
      [[maybe_unused]] const std::chrono::time_point<std::chrono::system_clock>&
          fallbackProfileStartTime) override {}

  static XpuptiScopeProfilerConfig& get(const Config& cfg) {
    return dynamic_cast<XpuptiScopeProfilerConfig&>(
        cfg.feature(kXpuptiProfilerConfigName));
  }

  Config& parent() const {
    return *parent_;
  }

  std::vector<std::string> activitiesXpuptiMetrics() const {
    return activitiesXpuptiMetrics_;
  }

  bool xpuptiProfilerPerKernel() const {
    return xpuptiProfilerPerKernel_;
  }

  int64_t xpuptiProfilerMaxScopes() const {
    return xpuptiProfilerMaxScopes_;
  }

  void setClientDefaults() override {
    setDefaults();
  }

  void printActivityProfilerConfig(std::ostream& s) const override;
  void setActivityDependentConfig() override {}
  static void registerFactory();

 protected:
  AbstractConfig* cloneDerived(AbstractConfig& parent) const override {
    XpuptiScopeProfilerConfig* clone = new XpuptiScopeProfilerConfig(*this);
    clone->parent_ = dynamic_cast<Config*>(&parent);
    return clone;
  }

 private:
  XpuptiScopeProfilerConfig() = delete;
  explicit XpuptiScopeProfilerConfig(Config& parent) : parent_(&parent) {}
  explicit XpuptiScopeProfilerConfig(const XpuptiScopeProfilerConfig& other) =
      default;

  // some defaults will depend on other configuration
  void setDefaults();

  // Associated Config object
  Config* parent_;

  // Counter metrics exposed via XPUPTI Profiler API
  std::vector<std::string> activitiesXpuptiMetrics_;

  // Collect profiler metrics per kernel - autoscope made
  bool xpuptiProfilerPerKernel_{false};

  // max number of scopes to configure the profiler for.
  // this has to be set before hand to reserve space for the output
  int64_t xpuptiProfilerMaxScopes_ = 0;
};

} // namespace KINETO_NAMESPACE
