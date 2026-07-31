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
#include <thread>
#include <vector>

#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "IActivityProfiler.h"

namespace libkineto {

class ActivityProfilerController;
struct CpuTraceBuffer;
class Config;

class ActivityProfilerInterface {
 public:
  virtual ~ActivityProfilerInterface() = default;

  virtual void init() {}
  virtual bool isInitialized() {
    return false;
  }
  virtual bool isActive() {
    return false;
  }
  virtual bool isStopped() const {
    return false;
  }

  // *** Asynchronous API ***
  // Instead of starting and stopping the trace manually, provide a start time
  // and duration and / or iteration stop criterion.
  // Tracing terminates when either condition is met.
  virtual void scheduleTrace([[maybe_unused]] const std::string& configStr) {}

  // Re-evaluate internal state to allow for triggering operations based
  // on number of iteration. each implicitly increments the iteration count
  virtual void step() {}

  // *** Synchronous API ***
  // These must be called in order:
  // prepareTrace -> startTrace -> stopTrace.

  // Many tracing structures are lazily initialized during trace collection,
  // with potentially high overhead.
  // Call prepareTrace to enable tracing, then run the region to trace
  // at least once (and ideally run the same code that is to be traced) to
  // allow tracing structures to be initialized.
  virtual void prepareTrace(
      [[maybe_unused]] const std::set<ActivityType>& activityTypes,
      [[maybe_unused]] const std::string& configStr = "") {}

  // Toggle GPU tracing as a trace is running to omit certain parts of a graph
  virtual void toggleCollectionDynamic([[maybe_unused]] const bool enable) {}

  // Start recording, potentially reusing any buffers allocated since
  // prepareTrace was called.
  virtual void startTrace() {}

  // Stop and process trace, producing an in-memory list of trace records.
  // The processing will be done synchronously (using the calling thread.)
  virtual std::unique_ptr<ActivityTraceInterface> stopTrace() {
    return nullptr;
  }

  // *** TraceActivity API ***
  // FIXME: Pass activityProfiler interface into clientInterface?
  virtual void pushCorrelationId([[maybe_unused]] uint64_t id) {}
  virtual void popCorrelationId() {}
  virtual void transferCpuTrace(
      [[maybe_unused]] std::unique_ptr<CpuTraceBuffer> traceBuffer) {}

  // Correlation ids for user defined spans
  virtual void pushUserCorrelationId([[maybe_unused]] uint64_t id) {}
  virtual void popUserCorrelationId() {}

  // Saves information for the current thread to be used in profiler output
  // Client must record any new kernel thread where the activity has occured.
  virtual void recordThreadInfo() {}

  // Record trace metadata, currently supporting only string key and values,
  // values with the same key are overwritten
  virtual void addMetadata(
      const std::string& key,
      const std::string& value) = 0;

  // Add a child activity profiler, this enables frameworks in the application
  // to enable custom framework events.
  virtual void addChildActivityProfiler(
      [[maybe_unused]] std::unique_ptr<IActivityProfiler> profiler) {}

  // Log Invariant Violation to factories enabled. This helps record
  // instances when the profiler behaves unexpectedly.
  virtual void logInvariantViolation(
      [[maybe_unused]] const std::string& profile_id,
      [[maybe_unused]] const std::string& assertion,
      [[maybe_unused]] const std::string& error,
      [[maybe_unused]] const std::string& group_profile_id = "") {}
};

} // namespace libkineto
