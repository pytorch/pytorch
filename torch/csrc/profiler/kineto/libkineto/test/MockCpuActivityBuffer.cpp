/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/MockCpuActivityBuffer.h"

#include "include/GenericTraceActivity.h"
#include "include/ThreadUtil.h"

namespace libkineto::test {

const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}

MockCpuActivityBuffer::MockCpuActivityBuffer(
    int64_t startTime,
    int64_t endTime) {
  span = TraceSpan(startTime, endTime, "Test trace");
  gpuOpCount = 0;
}

void MockCpuActivityBuffer::addOp(
    const std::string& name,
    int64_t startTime,
    int64_t endTime,
    int32_t correlation,
    const std::unordered_map<std::string, std::string>& metadataMap) {
  GenericTraceActivity op(span, ActivityType::CPU_OP, name);
  op.startTime = startTime;
  op.endTime = endTime;
  op.device = systemThreadId();
  op.resource = systemThreadId();
  op.id = correlation;

  for (const auto& [key, val] : metadataMap) {
    op.addMetadata(key, val);
  }

  emplace_activity(std::move(op));
  span.opCount++;
}

} // namespace libkineto::test
