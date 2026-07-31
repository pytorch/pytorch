/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <ostream>
#include <ratio>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityBuffers.h"
#include "GenericTraceActivity.h"
#include "output_base.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {
// Previous declaration of TraceSpan is struct. Must match the same here.
struct TraceSpan;
} // namespace KINETO_NAMESPACE

namespace KINETO_NAMESPACE {

class ArgsBuilder;
class Config;

struct pgConfig {
  pgConfig() = default;
  std::string pg_name;
  std::string pg_desc;
  std::string backend_config;
  std::string pg_size;
  std::string ranks;
};

struct DistributedInfo {
  DistributedInfo() = default;

  std::string backend;
  std::string rank;
  std::string world_size;
  std::string pg_count;
  std::string nccl_version;
  bool distInfo_present_{false};
};

class ChromeTraceLogger : public libkineto::ActivityLogger {
 public:
  explicit ChromeTraceLogger(const std::string& traceFileName);

  // Note: the caller of these functions should handle concurrency
  // i.e., we these functions are not thread-safe
  void handleDeviceInfo(const DeviceInfo& info, int64_t time) override;

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override;

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override;

  void handleTraceSpan(const TraceSpan& span) override;

  void handleActivity(const ITraceActivity& activity) override;
  void handleGenericActivity(const GenericTraceActivity& activity) override;

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata,
      const std::string& device_properties) override;

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime) override;

  void finalizeMemoryTrace(const std::string&, const Config&) override;

  std::string traceFileName() const {
    return fileName_;
  }

 protected:
  void finalizeTrace(int64_t endTime);

 private:
  // Create a flow event (arrow)
  void handleLink(
      char type,
      const ITraceActivity& e,
      int64_t id,
      const std::string& name);

  void addIterationMarker(const TraceSpan& span);

  void openTraceFile();

  void handleGenericInstantEvent(const ITraceActivity& op);

  void handleCounterEvent(const ITraceActivity& op);

  void handleGenericLink(const ITraceActivity& activity);

  void metadataToJSON(
      const std::unordered_map<std::string, std::string>& metadata);

  std::unordered_map<std::string, std::string> addEnvVarsToMetadata(
      const std::unordered_map<std::string, std::string>& metadata);

  void addOnDemandDistMetadata();

  // Chrome Trace event writer helpers.
  // The string_view pid/tid variants are the canonical implementations.
  // Integer overloads are provided for convenience at call sites with
  // numeric process/thread IDs.
  void writeMetadataEvent(
      std::string_view name,
      int64_t ts,
      std::string_view pid,
      std::string_view tid,
      std::string_view arg_key,
      std::string_view arg_value);

  void writeMetadataEvent(
      std::string_view name,
      int64_t ts,
      int64_t pid,
      int64_t tid,
      std::string_view arg_key,
      std::string_view arg_value);

  void writeCompleteEvent(
      std::string_view cat,
      std::string_view name,
      std::string_view pid,
      std::string_view tid,
      int64_t ts,
      int64_t dur,
      const ArgsBuilder& args);

  void writeCompleteEvent(
      std::string_view cat,
      std::string_view name,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      int64_t dur,
      const ArgsBuilder& args);

  void writeInstantEvent(
      std::string_view cat,
      std::string_view name,
      std::string_view scope,
      std::string_view pid,
      std::string_view tid,
      int64_t ts,
      const ArgsBuilder& args,
      bool finalEvent = false);

  void writeInstantEvent(
      std::string_view cat,
      std::string_view name,
      std::string_view scope,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      const ArgsBuilder& args,
      bool finalEvent = false);

  void writeCounterEvent(
      std::string_view cat,
      std::string_view name,
      std::string_view pid,
      std::string_view tid,
      int64_t ts,
      const ArgsBuilder& args);

  void writeCounterEvent(
      std::string_view cat,
      std::string_view name,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      const ArgsBuilder& args);

  void writeFlowEvent(
      char type,
      int64_t id,
      std::string_view pid,
      std::string_view tid,
      int64_t ts,
      std::string_view cat,
      std::string_view name);

  void writeFlowEvent(
      char type,
      int64_t id,
      int64_t pid,
      int64_t tid,
      int64_t ts,
      std::string_view cat,
      std::string_view name);

  // Copy the collective args (name, message sizes, dtype, process group, ranks,
  // seq, ...) recorded on a record_param_comms op into args. Backend-agnostic.
  void appendCollectiveArgs(
      ArgsBuilder& args,
      const ITraceActivity& collectiveRecord);

  // Enrich a device collective row from its linked record_param_comms op: copy
  // the collective args and fold its process group into the trace's
  // distributedInfo under the given backend labels.
  void appendCollectiveMetadata(
      ArgsBuilder& args,
      const ITraceActivity& collectiveRecord,
      const std::string& backend,
      const std::string& backendConfig);

  std::string fileName_;
  std::string tempFileName_;
  std::ofstream traceOf_;
  DistributedInfo distInfo_ = DistributedInfo();
  // Map of all observed process groups to their configs in trace. Key is
  // pg_name, value is pgConfig that will be used to populate pg_config in
  // distributedInfo of trace
  std::unordered_map<std::string, pgConfig> pgMap_ = {};

  // Offset added to stream ID for CUDA_SYNC events to place them on a
  // separate row from kernel events in the Chrome Trace JSON output.
  static constexpr int64_t kSyncStreamTidOffset = 1000000;

  // Tracks which (device, virtualTid) pairs have had thread_name metadata
  // emitted, to avoid duplicates.
  std::unordered_set<int64_t> syncStreamMetadataEmitted_;
};

// std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
#define _KINETO_GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
#define _KINETO_GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
#define _KINETO_GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

// There are tools like Chrome Trace Viewer that uses double to represent
// each element in the timeline. Double has a 53 bit mantissa to support
// up to 2^53 significant digits (up to 9007199254740992). This holds at the
// nanosecond level, about 3 months and 12 days. So, let's round base time to
// 3 months intervals, so we can still collect traces across ranks relative
// to each other.
// A month is 2629746, so 3 months is 7889238.
using _trimonths =
    std::chrono::duration<_KINETO_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
#undef _GLIBCXX_CHRONO_INT64_T

class ChromeTraceBaseTime {
 public:
  ChromeTraceBaseTime() = default;
  static ChromeTraceBaseTime& singleton();
  void init() {
    get();
  }
  int64_t get() {
    // Make all timestamps relative to 3 month intervals.
    static int64_t base_time = libkineto::timeSinceEpoch(
        std::chrono::time_point<std::chrono::system_clock>(
            std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
    return base_time;
  }
};

int64_t transToRelativeTime(int64_t time);

} // namespace KINETO_NAMESPACE
