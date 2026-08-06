/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "Config.h"
#include "GenericTraceActivity.h"

/* This file includes an abstract base class for an activity profiler
 * that can be implemented by multiple tracing agents in the application.
 * The high level Kineto profiler can co-ordinate start and end of tracing
 * and combine together events from multiple such activity profilers.
 */

namespace libkineto {

struct CpuTraceBuffer;

#ifdef _MSC_VER
// workaround for the predefined ERROR macro on Windows
#undef ERROR
#endif // _MSC_VER

enum class TraceStatus {
  READY, // Accepting trace requests
  WARMUP, // Performing trace warmup
  RECORDING, // Actively collecting activities
  PROCESSING, // Recording is complete, preparing results
  ERROR, // One or more errors (and possibly also warnings) occurred.
  WARNING, // One or more warnings occurred.
};

/* DeviceInfo:
 *   Can be used to specify process name, sort order, PID and device label.
 *   The sort order is determined by the sortIndex field to handle ordering of
 *   processes and gpu rows in the trace viewer.
 */
struct DeviceInfo {
  int64_t id; // process id
  int64_t sortIndex; // position in trace view
  const std::string name; // process name
  const std::string label; // device label
};

/* ResourceInfo:
 *   Can be used to specify resource inside device
 */
struct ResourceInfo {
  int64_t id; // resource id
  int64_t sortIndex; // position in trace view
  int64_t deviceId; // id of device which owns this resource (specified in
                    // DeviceInfo.id)
  const std::string name; // resource name
};

using getLinkedActivityCallback = std::function<const ITraceActivity*(int32_t)>;

/* IActivityProfilerSession:
 *   an opaque object that can be used by a high level profiler to
 *   start/stop and return trace events.
 */
class IActivityProfilerSession {
 public:
  virtual ~IActivityProfilerSession() = default;

  // start the trace collection synchronously
  virtual void start() = 0;

  // stop the trace collection synchronously
  virtual void stop() = 0;

  TraceStatus status() {
    return status_;
  }

  // returns errors with this trace
  virtual std::vector<std::string> errors() = 0;

  // processes trace activities using logger
  virtual void processTrace(ActivityLogger& logger) = 0;

  virtual void processTrace(
      ActivityLogger& logger,
      [[maybe_unused]] getLinkedActivityCallback getLinkedActivity,
      [[maybe_unused]] int64_t startTime,
      [[maybe_unused]] int64_t endTime) {
    processTrace(logger);
  }

  // returns device info used in this trace, could be nullptr
  virtual std::unique_ptr<DeviceInfo> getDeviceInfo() = 0;

  // returns resource info used in this trace, could be empty
  virtual std::vector<ResourceInfo> getResourceInfos() = 0;

  // release ownership of the trace events and metadata
  virtual std::unique_ptr<CpuTraceBuffer> getTraceBuffer() = 0;

  // XXX define trace formats
  // virtual save(string name, TraceFormat format)

  virtual void pushCorrelationId([[maybe_unused]] uint64_t id) {}
  virtual void popCorrelationId() {}

  virtual void pushUserCorrelationId([[maybe_unused]] uint64_t id) {}
  virtual void popUserCorrelationId() {}

  virtual void toggleCollectionDynamic([[maybe_unused]] bool enable) {}

  virtual std::string getDeviceProperties() {
    return "";
  }

  virtual std::unordered_map<std::string, std::string> getMetadata() {
    return {};
  }

 protected:
  TraceStatus status_ = TraceStatus::READY;
};

/* Activity Profiler Plugins:
 *   These allow other frameworks to integrate into Kineto's primariy
 *   activity profiler. While the primary activity profiler handles
 *   timing the trace collections and correlating events the plugins
 *   can become source of new trace activity types.
 */
class IActivityProfiler {
 public:
  virtual ~IActivityProfiler() = default;

  // name of profiler
  [[nodiscard]] virtual const std::string& name() const = 0;

  // returns activity types this profiler supports
  [[nodiscard]] virtual const std::set<ActivityType>& availableActivities()
      const = 0;

  // Calls prepare() on registered tracer providers passing in the relevant
  // activity types. Returns a profiler session handle
  virtual std::unique_ptr<IActivityProfilerSession> configure(
      const std::set<ActivityType>& activity_types,
      const Config& config) = 0;

  // asynchronous version of the above with future timestamp and duration.
  virtual std::unique_ptr<IActivityProfilerSession> configure(
      int64_t ts_ms,
      int64_t duration_ms,
      const std::set<ActivityType>& activity_types,
      const Config& config) = 0;
};

} // namespace libkineto
