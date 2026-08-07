/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityBuffers.h"
#include "Config.h"
#include "GenericTraceActivity.h"
#include "Logger.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

class Config;

class MemoryTraceLogger : public ActivityLogger {
 public:
  MemoryTraceLogger(const Config& config) : config_(config.clone()) {
    activities_.reserve(100000);
  }

  // Note: the caller of these functions should handle concurrency
  // i.e., these functions are not thread-safe
  void handleDeviceInfo(const DeviceInfo& info, int64_t time) override {
    deviceInfoList_.emplace_back(info, time);
  }

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override {
    resourceInfoList_.emplace_back(info, time);
  }

  void handleOverheadInfo(
      [[maybe_unused]] const OverheadInfo& info,
      [[maybe_unused]] int64_t time) override {}

  void handleTraceSpan([[maybe_unused]] const TraceSpan& span) override {
    // Handled separately
  }

  template <class T>
  void addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    activities_.push_back(wrappers_.back().get());
  }

  // Just add the pointer to the list - ownership of the underlying
  // objects must be transferred in ActivityBuffers via finalizeTrace
  void handleActivity(const ITraceActivity& activity) override {
    activities_.push_back(&activity);
  }
  void handleGenericActivity(const GenericTraceActivity& activity) override {
    addActivityWrapper(activity);
  }

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override {
    metadata_ = metadata;
    device_properties_ = device_properties;
  }

  void finalizeTrace(
      [[maybe_unused]] const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override {
    buffers_ = std::move(buffers);
    endTime_ = endTime;
  }

  void finalizeMemoryTrace(const std::string&, const Config&) override {
    LOG(INFO) << "finalizeMemoryTrace not implemented for MemLogger";
  }

  const std::vector<const ITraceActivity*>* traceActivities() {
    return &activities_;
  }

  void log(ActivityLogger& logger) {
    logger.handleTraceStart(metadata_, device_properties_);
    for (auto& p : deviceInfoList_) {
      logger.handleDeviceInfo(p.first, p.second);
    }
    for (auto& p : resourceInfoList_) {
      logger.handleResourceInfo(p.first, p.second);
    }
    for (auto& activity : activities_) {
      activity->log(logger);
    }
    for (auto& cpu_trace_buffer : buffers_->cpu) {
      logger.handleTraceSpan(cpu_trace_buffer->span);
    }
    // Hold on to the buffers
    logger.finalizeTrace(*config_, nullptr, endTime_);
  }

  void setChromeLogger(std::shared_ptr<ActivityLogger> logger) {
    chrome_logger_ = std::move(logger);
  }

  std::shared_ptr<ActivityLogger> getChromeLogger() {
    return chrome_logger_;
  }

 private:
  std::unique_ptr<Config> config_;
  // Optimization: Remove unique_ptr by keeping separate vector per type
  std::vector<const ITraceActivity*> activities_;
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
  std::vector<std::pair<DeviceInfo, int64_t>> deviceInfoList_;
  std::vector<std::pair<ResourceInfo, int64_t>> resourceInfoList_;
  std::unique_ptr<ActivityBuffers> buffers_;
  std::unordered_map<std::string, std::string> metadata_;
  std::string device_properties_;
  int64_t endTime_{0};
  std::shared_ptr<ActivityLogger> chrome_logger_;
};

} // namespace KINETO_NAMESPACE
