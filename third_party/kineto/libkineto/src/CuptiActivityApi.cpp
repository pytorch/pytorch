/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiActivityApi.h"

#include <chrono>
#include <mutex>
#include <thread>

#include "DeviceUtil.h"
#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

// Set to 4MB to avoid constantly creating buffers (especially for networks
// that have many small memcpy such as sparseNN). CUPTI recommends between
// 1MB to 10MB.
// Given the kDefaultActivitiesMaxGpuBufferSize is around 128MB, in the worst
// case, there will be 32 buffers contending for the mutex.
constexpr size_t kBufSize(4 * 1024 * 1024);

inline bool cuptiTearDown_() {
  auto teardown_env = getenv("TEARDOWN_CUPTI");
  return teardown_env != nullptr && strcmp(teardown_env, "1") == 0;
}

inline bool cuptiLazyInit_() {
  return cuptiTearDown_() && getenv("DISABLE_CUPTI_LAZY_REINIT") == nullptr;
}

inline void reenableCuptiCallbacks_(CuptiCallbackApi& cbapi) {
  // Re-enable callbacks from the past if they exist.
  LOG(INFO) << "Re-enable previous CUPTI callbacks - Starting";
  VLOG(1) << "  CUPTI subscriber before reinit:" << cbapi.getCuptiSubscriber();
  cbapi.initCallbackApi();
  if (cbapi.initSuccess()) {
    VLOG(1) << "  CUPTI subscriber after reinit:" << cbapi.getCuptiSubscriber();
    bool status = cbapi.reenableCallbacks();
    if (!status) {
      LOG(WARNING)
          << "Re-enable previous CUPTI callbacks - Failed to reenableCallbacks";
    } else {
      LOG(INFO) << "Re-enable previous CUPTI callbacks - Successful";
    }
  } else {
    LOG(WARNING)
        << "Re-enable previous CUPTI callbacks - Failed to initCallbackApi";
  }
}

CuptiActivityApi& CuptiActivityApi::singleton() {
  static auto* instance = new CuptiActivityApi();
  return *instance;
}

void CuptiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_.load(
          std::memory_order_relaxed)) {
    return;
  }
  VLOG(2) << "pushCorrelationID(" << id << ")";
  switch (type) {
    case Default:
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id));
      break;
    case User:
      CUPTI_CALL(cuptiActivityPushExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, id));
  }
}

void CuptiActivityApi::popCorrelationID(CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_.load(
          std::memory_order_relaxed)) {
    return;
  }
  switch (type) {
    case Default:
      CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr));
      break;
    case User:
      CUPTI_CALL(cuptiActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, nullptr));
  }
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    CUpti_Activity*& record) {
  CUptiResult status = CUPTI_CALL_NOWARN(
      cuptiActivityGetNextRecord(buffer, valid_size, &record));
  if (status != CUPTI_SUCCESS) {
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      CUPTI_CALL(status);
    }
    record = nullptr;
  }
  return record != nullptr;
}

void CuptiActivityApi::setMaxBufferSize(int64_t size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

void CuptiActivityApi::setDeviceBufferSize(size_t size) {
  size_t valueSize = sizeof(size_t);
  CUPTI_CALL(cuptiActivitySetAttribute(
      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, &size));
}

void CuptiActivityApi::setDeviceBufferPoolLimit(size_t limit) {
  size_t valueSize = sizeof(size_t);
  CUPTI_CALL(cuptiActivitySetAttribute(
      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &valueSize, &limit));
}

void CuptiActivityApi::forceLoadCupti() {
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

void CuptiActivityApi::preConfigureCUPTI() {
  if (!isGpuAvailable()) {
    return;
  }
}

void CUPTIAPI CuptiActivityApi::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void CuptiActivityApi::bufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  std::lock_guard<std::mutex> guard(mutex_);
  LOG(VERBOSE) << "CUPTI buffer requested";
  if (static_cast<int64_t>(allocatedGpuTraceBuffers_.size()) >=
      maxGpuBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max GPU buffer count ("
                 << allocatedGpuTraceBuffers_.size()
                 << " >= " << maxGpuBufferCount_ << ") - terminating tracing";
    // Return null buffer to CUPTI. Per the CUPTI documentation for
    // CUpti_BuffersCallbackRequestFunc: "If set to NULL then no buffer is
    // returned." CUPTI will drop activity records, which are counted by
    // cuptiActivityGetNumDroppedRecords.
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }

  auto buf = std::make_unique<CuptiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGpuTraceBuffers_[*buffer] = std::move(buf);

  *maxNumRecords = 0;
}

std::unique_ptr<CuptiActivityBufferMap> CuptiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      if (readyGpuTraceBuffers_) {
        return std::move(readyGpuTraceBuffers_);
      }
      return nullptr;
    }
  }

  VLOG(1) << "Flushing GPU activity buffers";
  time_point<system_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = system_clock::now();
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcupti and mutex_ is acquired there.
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(system_clock::now() - t1).count();
  }

#if (CUDART_VERSION >= 12030)
  // Clear out per-thread buffer flag in case it was set
  uint8_t value = 0;
  size_t sizeof_value = sizeof(value);

  CUPTI_CALL(cuptiActivitySetAttribute(
      CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, &sizeof_value, &value));
#endif // (CUDART_VERSION >= 12030)
  std::lock_guard<std::mutex> guard(mutex_);
  // Transfer ownership of buffers to caller. A new map is created on-demand.
  return std::move(readyGpuTraceBuffers_);
}

int CuptiActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    const std::function<void(const CUpti_Activity*)>& handler) {
  int count = 0;
  if (buf && validSize) {
    CUpti_Activity* record{nullptr};
    while ((nextActivityRecord(buf, validSize, record))) {
      handler(record);
      ++count;
    }
  }
  return count;
}

const std::pair<int, size_t> CuptiActivityApi::processActivities(
    CuptiActivityBufferMap& buffers,
    const std::function<void(const CUpti_Activity*)>& handler) {
  std::pair<int, size_t> res{0, 0};
  for (auto& pair : buffers) {
    // No lock needed - only accessed from this thread
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
  return res;
}

void CuptiActivityApi::flushActivities() {
  CUPTI_CALL(cuptiActivityFlushAll(0));
}

void CuptiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcupti and mutex_ is acquired there.
  CUPTI_CALL(cuptiActivityFlushAll(0));
  // FIXME: We might want to make sure we reuse
  // the same memory during warmup and tracing.
  // Also, try to use the amount of memory required
  // for active tracing during warmup.
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  readyGpuTraceBuffers_ = nullptr;
}

void CUPTIAPI CuptiActivityApi::bufferCompletedTrampoline(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  singleton().bufferCompleted(ctx, streamId, buffer, 0, validSize);
}

void CuptiActivityApi::bufferCompleted(
    CUcontext ctx,
    uint32_t streamId,
    uint8_t* buffer,
    size_t /* unused */,
    size_t validSize) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = allocatedGpuTraceBuffers_.find(buffer);
    if (it == allocatedGpuTraceBuffers_.end()) {
      LOG(ERROR) << "bufferCompleted called with unknown buffer: "
                 << static_cast<void*>(buffer);
      return;
    }

    if (!readyGpuTraceBuffers_) {
      readyGpuTraceBuffers_ = std::make_unique<CuptiActivityBufferMap>();
    }
    // Set valid size of buffer before moving to ready map
    it->second->setSize(validSize);
    (*readyGpuTraceBuffers_)[it->first] = std::move(it->second);
    allocatedGpuTraceBuffers_.erase(it);
  }

  // report any records dropped from the queue; to avoid unnecessary cupti
  // API calls, we make it report only in verbose mode (it doesn't happen
  // often in our testing anyways)
  // Can't hold mutex_ during this call, since cuptiActivityGetNumDroppedRecords
  // re-enters CUPTI and can acquire CUPTI's internal lock, while CUPTI
  // callbacks (e.g. bufferRequested) acquire mutex_ under that same lock -
  // causing an ABBA deadlock in multi-threaded environments (e.g. free-threaded
  // Python).
  if (VLOG_IS_ON(1)) {
    size_t dropped = 0;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
}

void CuptiActivityApi::enableCuptiActivities(
    const std::set<ActivityType>& selected_activities,
    bool enablePerThreadBuffers) {
  // Lazily support re-init of CUPTI Callbacks, if they were finalized before.
  auto& cbapi = CuptiCallbackApi::singleton();
  if ((tracingEnabled_ == 0u) && !cbapi.initSuccess() && cuptiLazyInit_()) {
    reenableCuptiCallbacks_(cbapi);
  }

  if (enablePerThreadBuffers) {
#if (CUDART_VERSION >= 12030)
    uint8_t value = 1;
    size_t sizeof_value = sizeof(value);
    LOG(INFO) << ("Enabling per-thread activity buffer");
    CUPTI_CALL(cuptiActivitySetAttribute(
        CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, &sizeof_value, &value));
#else
    LOG(WARNING) << "Per-thread activity buffer is not supported on CUDA";
#endif // (CUDART_VERSION >= 12030)
  } else {
    LOG(VERBOSE) << ("Not enabling per-thread activity buffer");
  }

  CUPTI_CALL(cuptiActivityRegisterCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_.store(false, std::memory_order_relaxed);
  using enum ActivityType;
  for (const auto& activity : selected_activities) {
    if (activity == GPU_MEMCPY) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == GPU_MEMSET) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    }
    if (activity == CONCURRENT_KERNEL) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }
    if (activity == EXTERNAL_CORRELATION) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_.store(true, std::memory_order_relaxed);
    }
    if (activity == CUDA_SYNC) {
#if CUDA_VERSION >= 13000
      CUPTI_CALL(cuptiActivityEnableCudaEventDeviceTimestamps(true));
#endif
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
    }
    if (activity == CUDA_RUNTIME) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
#if (CUDART_VERSION >= 12050)
      CUPTI_CALL(cuptiActivityEnableRuntimeApi(
          CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020, 0));
#endif
    }
    if (activity == CUDA_DRIVER) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
#if (CUDART_VERSION >= 12050)
      CUPTI_CALL(cuptiActivityEnableDriverApi(
          CUPTI_DRIVER_TRACE_CBID_cuKernelGetAttribute, 0));
      CUPTI_CALL(cuptiActivityEnableDriverApi(
          CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxGetState, 0));
      CUPTI_CALL(cuptiActivityEnableDriverApi(
          CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent, 0));
#endif
    }
    if (activity == OVERHEAD) {
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    }
  }

  tracingEnabled_ = 1;

  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void CuptiActivityApi::disableCuptiActivities(
    const std::set<ActivityType>& selected_activities) {
  using enum ActivityType;
  for (const auto& activity : selected_activities) {
    if (activity == GPU_MEMCPY) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    }
    if (activity == GPU_MEMSET) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
    }
    if (activity == CONCURRENT_KERNEL) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }
    if (activity == EXTERNAL_CORRELATION) {
      CUPTI_CALL(
          cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    }
    if (activity == CUDA_SYNC) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
    }
    if (activity == CUDA_RUNTIME) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    }
    if (activity == CUDA_DRIVER) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    }
    if (activity == OVERHEAD) {
      CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    }
  }
  externalCorrelationEnabled_.store(false, std::memory_order_relaxed);
}

void CuptiActivityApi::teardownContext() {
  if (!tracingEnabled_) {
    return;
  }
  if (tearingDown_) {
    return;
  }
  if (cuptiTearDown_()) {
    tearingDown_ = 1;
    LOG(INFO) << "teardownCupti starting";

    // PyTorch Profiler is synchronous, so teardown needs to be run async in
    // this thread.
    std::thread teardownThread([&] {
      auto& cbapi = CuptiCallbackApi::singleton();
      if (!cbapi.initSuccess()) {
        cbapi.initCallbackApi();
        if (!cbapi.initSuccess()) {
          LOG(WARNING) << "CUPTI Callback failed to init, skipping teardown";
          tearingDown_ = 0;
          return;
        }
      }
      // Subscribe callbacks to call cuptiFinalize in the exit callback of these
      // APIs
      bool status = cbapi.enableCallbackDomain(CUPTI_CB_DOMAIN_RUNTIME_API);
      status = status && cbapi.enableCallbackDomain(CUPTI_CB_DOMAIN_DRIVER_API);
      if (!status) {
        LOG(WARNING)
            << "CUPTI Callback failed to enable for domain, skipping teardown";
        tearingDown_ = 0;
        return;
      }

      // Force Flush before finalize
      CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

      LOG(INFO) << "  CUPTI subscriber before finalize:"
                << cbapi.getCuptiSubscriber();
      teardownCupti_ = 1;
      std::unique_lock<std::mutex> lck(finalizeMutex_);
      finalizeCond_.wait(lck, [&] { return teardownCupti_ == 0; });
      lck.unlock();
      LOG(INFO) << "teardownCupti complete";

      teardownCupti_ = 0;
      tracingEnabled_ = 0;

      // Remove the callbacks used specifically for cuptiFinalize
      cbapi.disableCallbackDomain(CUPTI_CB_DOMAIN_RUNTIME_API);
      cbapi.disableCallbackDomain(CUPTI_CB_DOMAIN_DRIVER_API);

      // Re-init CUPTI Callbacks if Lazy Re-init is not enabled.
      if (!cuptiLazyInit_()) {
        reenableCuptiCallbacks_(cbapi);
      }
      tearingDown_ = 0;
    });
    teardownThread.detach();
  }
}

} // namespace KINETO_NAMESPACE
