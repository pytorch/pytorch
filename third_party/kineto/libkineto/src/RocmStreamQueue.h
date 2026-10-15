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

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

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

// Map a HIP memcpy kind to the rocprofiler copy operation used to render
// copy direction strings.  Copies issued through APIs that take no kind
// argument (hipMemcpyPeerAsync, hipMemcpyDtoD*, ...) always report
// hipMemcpyDefault, so for those the direction is derived from the API id.
// ROCPROFILER_MEMORY_COPY_NONE is returned when neither source tells us.
inline uint32_t copyOperationFromApi(hipMemcpyKind kind, uint32_t apiId) {
  switch (kind) {
    case hipMemcpyHostToHost:
      return ROCPROFILER_MEMORY_COPY_HOST_TO_HOST;
    case hipMemcpyHostToDevice:
      return ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
    case hipMemcpyDeviceToHost:
      return ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST;
    case hipMemcpyDeviceToDevice:
    case hipMemcpyDeviceToDeviceNoCU:
      return ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE;
    default:
      break;
  }

  switch (apiId) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
      return ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
      return ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
      return ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST;
    default:
      return ROCPROFILER_MEMORY_COPY_NONE;
  }
}

// Classify kernel dispatches the runtime issued to implement HIP memset or
// memcpy API calls.  Some memsets/copies are implemented with a fill/copy
// kernel instead of SDMA; for those, the dispatch record is the *only*
// GPU-side artifact (no MEMORY_SET/MEMORY_COPY record is emitted), so there
// is no double counting when it is reported as the memset/memcpy the user
// asked for (CUPTI parity) rather than as a user kernel.  The dispatch
// shares the correlation id of the runtime API record.
inline void classifyRuntimeInternalDispatches(
    std::vector<rocprofBase*>& rows,
    const std::unordered_map<uint64_t, size_t>& memsetOps) {
  std::unordered_map<uint64_t, const rocprofCopyRow*> copyByCorrelation;
  for (const auto* item : rows) {
    if (item->type == ROCTRACER_ACTIVITY_COPY) {
      const auto* copy = reinterpret_cast<const rocprofCopyRow*>(item);
      copyByCorrelation.emplace(copy->id, copy);
    }
  }
  if (memsetOps.empty() && copyByCorrelation.empty()) {
    return;
  }
  for (auto* item : rows) {
    if (item->type != ROCTRACER_ACTIVITY_ASYNC) {
      continue;
    }
    auto* async = reinterpret_cast<rocprofAsyncRow*>(item);
    if (async->domain != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH ||
        async->internalApi != RocLogger::RuntimeInternalNone) {
      continue;
    }
    const auto copyIt = copyByCorrelation.find(async->id);
    if (copyIt != copyByCorrelation.end()) {
      async->internalApi = RocLogger::RuntimeInternalMemcpy;
      async->internalBytes = copyIt->second->size;
      async->op =
          copyOperationFromApi(copyIt->second->kind, copyIt->second->cid);
      continue;
    }
    const auto memsetIt = memsetOps.find(async->id);
    if (memsetIt != memsetOps.end()) {
      async->internalApi = RocLogger::RuntimeInternalMemset;
      async->internalBytes = memsetIt->second;
    }
  }
}

} // namespace detail
} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
