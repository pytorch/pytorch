/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <thread>
#include <unordered_map>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

namespace KINETO_NAMESPACE {
struct ActivityBuffers;
}

namespace libkineto {

using namespace KINETO_NAMESPACE;

// Used by sortIndex to put GPU tracks at the bottom
// of the trace timelines. The largest valid CPU PID is 4,194,304,
// so 5000000 is enough to guarantee that GPU tracks are sorted after CPU.
constexpr int64_t kExceedMaxPid = 5000000;

class ActivityLogger {
 public:
  virtual ~ActivityLogger() = default;

  struct OverheadInfo {
    explicit OverheadInfo(const std::string& name) : name(name) {}
    const std::string name;
  };

  virtual void handleDeviceInfo(const DeviceInfo& info, int64_t time) = 0;

  virtual void handleResourceInfo(const ResourceInfo& info, int64_t time) = 0;

  virtual void handleOverheadInfo(const OverheadInfo& info, int64_t time) = 0;

  virtual void handleTraceSpan(const TraceSpan& span) = 0;

  virtual void handleActivity(const libkineto::ITraceActivity& activity) = 0;
  virtual void handleGenericActivity(
      const libkineto::GenericTraceActivity& activity) = 0;

  virtual void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) = 0;

  void handleTraceStart() {
    handleTraceStart(std::unordered_map<std::string, std::string>(), "");
  }

  virtual void finalizeMemoryTrace(const std::string&, const Config&) = 0;

  virtual void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) = 0;

 protected:
  ActivityLogger() = default;
};

} // namespace libkineto
