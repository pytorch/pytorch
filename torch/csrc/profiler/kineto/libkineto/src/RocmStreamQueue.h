/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef HAS_ROCTRACER

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "RocLogger.h"

namespace KINETO_NAMESPACE {
namespace detail {

inline uint64_t streamIdFromHipStream(hipStream_t stream) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
}

inline uint64_t runtimeStreamId(const rocprofBase* item) {
  if (item->type == ROCTRACER_ACTIVITY_KERNEL) {
    return streamIdFromHipStream(
        reinterpret_cast<const rocprofKernelRow*>(item)->stream);
  }
  if (item->type == ROCTRACER_ACTIVITY_COPY) {
    return streamIdFromHipStream(
        reinterpret_cast<const rocprofCopyRow*>(item)->stream);
  }
  return 0;
}

// Assign logical stream ID to async kernel and copy rows. Remap to small index
// for proper display. Recover logical stream ID from the correlated runtime
// row. isAsyncOp selects the async rows to process.
template <class IsAsyncOp>
void backfillAsyncStreams(
    std::vector<rocprofBase*>& rows,
    IsAsyncOp isAsyncOp) {
  // Map correlation id -> HIP stream from the runtime kernel/copy rows.
  std::unordered_map<uint64_t, uint64_t> streamByCorrelation;
  for (const auto* item : rows) {
    const uint64_t sid = runtimeStreamId(item);
    if (sid != 0) {
      streamByCorrelation[item->id] = sid;
    }
  }
  if (streamByCorrelation.empty()) {
    return;
  }

  // Set stream from the correlation-matched runtime row.
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (!isAsyncOp(*async) || async->stream != 0) {
      continue;
    }
    const auto it = streamByCorrelation.find(async->id);
    if (it != streamByCorrelation.end()) {
      async->stream = it->second;
    }
  }

  // Fallback: (e.g. ROCm-internal dispatches like __amd_rocclr_copyBuffer),
  // infer the stream from the HSA queue, given the queue maps directly to a HIP
  // stream
  std::unordered_map<uint64_t, uint64_t> streamByQueue;
  std::unordered_set<uint64_t> ambiguousQueues;
  for (const auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    const auto* async = reinterpret_cast<const rocprofAsyncRow*>(item);
    if (!isAsyncOp(*async) || async->stream == 0 || async->queue == 0) {
      continue;
    }
    if (ambiguousQueues.count(async->queue) > 0) {
      continue;
    }
    const auto [it, inserted] =
        streamByQueue.emplace(async->queue, async->stream);
    if (!inserted && it->second != async->stream) {
      streamByQueue.erase(it);
      ambiguousQueues.insert(async->queue);
    }
  }
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (!isAsyncOp(*async) || async->stream != 0) {
      continue;
    }
    const auto it = streamByQueue.find(async->queue);
    if (it != streamByQueue.end()) {
      async->stream = it->second;
    }
  }

  // Remap raw HIP stream pointers to small per-device indices, avoiding
  // misrenders of large 64-bit tid values. Fallback: if still no stream ID, use
  // HSA queue ID tagged with high bit to avoid collisions with stream IDs
  constexpr uint64_t kQueueKeyTag = uint64_t{1} << 63;
  std::unordered_map<int, std::unordered_map<uint64_t, uint64_t>>
      deviceStreamToIndex;
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (!isAsyncOp(*async)) {
      continue;
    }
    uint64_t key;
    if (async->stream != 0) {
      key = async->stream;
    } else if (async->queue != 0) {
      key = async->queue | kQueueKeyTag;
    } else {
      continue;
    }
    auto& mapping = deviceStreamToIndex[async->device];
    auto [it, inserted] = mapping.emplace(key, mapping.size() + 1);
    async->stream = it->second;
  }
}

} // namespace detail
} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
