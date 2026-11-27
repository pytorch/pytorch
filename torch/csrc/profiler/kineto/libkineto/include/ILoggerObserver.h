/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define NOGDI
#include <string>

// Stages in libkineto used when pushing logs to UST Logger.
// Emitted when a trace request is cancelled: rejected before it runs
// (busy/pending/can't-start) or an in-flight trace preempted by a
// higher-priority request.
constexpr char kCancellationStage[] = "Cancellation";
constexpr char kWarmUpStage[] = "Warm Up";
constexpr char kCollectionStage[] = "Collection";
constexpr char kPostProcessingStage[] = "Post Processing";

// Special string in UST for determining if traces are empty
constexpr char kEmptyTrace[] =
    "No Valid Trace Events (CPU/GPU) found. Outputting empty trace.";

#if !USE_GOOGLE_LOG

#include <map>
#include <vector>

#include <cstdint>

#ifdef _MSC_VER
// unset a predefined ERROR (windows)
#undef ERROR
#endif // _MSC_VER

namespace libkineto {

enum LoggerOutputType {
  VERBOSE = 0,
  INFO = 1,
  WARNING = 2,
  STAGE = 3,
  ERROR = 4,
  USDT = 5,
  ENUM_COUNT = 6
};

const char* toString(LoggerOutputType t);
LoggerOutputType toLoggerOutputType(const std::string& str);

constexpr int LoggerTypeCount = (int)LoggerOutputType::ENUM_COUNT;

class ILoggerObserver {
 public:
  virtual ~ILoggerObserver() = default;
  virtual void write(const std::string& message, LoggerOutputType ot) = 0;
  virtual const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() = 0;
  virtual void reset() = 0;
  virtual void addDevice(int64_t device) = 0;
  virtual void setTraceDurationMS(int64_t duration) = 0;
  virtual void addEventCount(int64_t count) = 0;
  virtual void setTraceID([[maybe_unused]] const std::string& traceID) {}
  virtual void setGroupTraceID(
      [[maybe_unused]] const std::string& groupTraceID) {}
  virtual void addDestination(const std::string& dest) = 0;
  virtual void setTriggerOnDemand() {}
  virtual void addMetadata(
      const std::string& key,
      const std::string& value) = 0;
  // Metadata that is constant for the process lifetime (e.g. GPU/driver
  // versions). Unlike addMetadata, this is NOT cleared by reset(), so it
  // survives across traces and must not be used for per-trace values.
  virtual void addPersistentMetadata(
      [[maybe_unused]] const std::string& key,
      [[maybe_unused]] const std::string& value) {}
  // Emit a standalone record that an on-demand trace request was cancelled,
  // attributed to the cancelled request's trace id. Stateless: does not read or
  // mutate the observer's per-trace state (a trace may be active).
  virtual void writeStageCancellation(
      [[maybe_unused]] const std::string& trace_id,
      [[maybe_unused]] const std::string& group_trace_id,
      [[maybe_unused]] const std::string& reason) {}
};

} // namespace libkineto

#endif // !USE_GOOGLE_LOG
