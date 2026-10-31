/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "include/libkineto.h"

// Shared fixtures for the CUDA and ROCm activity-profiler tests: a mock CPU-op
// buffer, a default trace span, and the param-comms / collective metadata
// field-name constants both tests assert on. These depend on the kineto
// activity types, so they live here rather than in the generic TestUtils.h.

namespace libkineto::test {

inline const std::string kParamCommsCallName = "record_param_comms";
inline constexpr auto kCollectiveName = "Collective name";
inline constexpr auto kDtype = "dtype";
inline constexpr auto kInMsgNelems = "In msg nelems";
inline constexpr auto kOutMsgNelems = "Out msg nelems";
inline constexpr auto kInSplit = "In split size";
inline constexpr auto kOutSplit = "Out split size";
inline constexpr auto kGroupSize = "Group size";
inline constexpr const char* kProcessGroupName = "Process Group Name";
inline constexpr const char* kProcessGroupDesc = "Process Group Description";
inline constexpr const char* kGroupRanks = "Process Group Ranks";
inline constexpr auto kSeqNum = "Seq";
inline constexpr const char* kCommsId = "Comms Id";
inline constexpr int32_t kTruncatLength = 30;

// A shared default TraceSpan used to seed test activities.
const TraceSpan& defaultTraceSpan();

// A CpuTraceBuffer with a convenience addOp() for building CPU-side ops.
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime);

  void addOp(
      const std::string& name,
      int64_t startTime,
      int64_t endTime,
      int32_t correlation,
      const std::unordered_map<std::string, std::string>& metadataMap = {});
};

} // namespace libkineto::test
