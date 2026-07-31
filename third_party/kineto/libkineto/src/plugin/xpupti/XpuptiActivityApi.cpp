/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityApi.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

#include "Logger.h"

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(4 * 1024 * 1024);

XpuptiActivityApi& XpuptiActivityApi::singleton() {
  static XpuptiActivityApi instance;
  return instance;
}

void XpuptiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      XPUPTI_CALL(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, id));
      break;
    case User:
      XPUPTI_CALL(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, id));
  }
}

void XpuptiActivityApi::popCorrelationID(CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      XPUPTI_CALL(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, nullptr));
      break;
    case User:
      XPUPTI_CALL(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, nullptr));
  }
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    pti_view_record_base*& record) {
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
  return record != nullptr;
}

void XpuptiActivityApi::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XpuptiActivityApi::bufferRequested(uint8_t** buffer, size_t* size) {
  std::lock_guard<std::mutex> guard(mutex_);

  auto buf = std::make_unique<XpuptiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGpuTraceBuffers_[*buffer] = std::move(buf);
}

std::unique_ptr<XpuptiActivityBufferMap> XpuptiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return nullptr;
    }
  }

  std::chrono::time_point<std::chrono::system_clock> t1;
  XPUPTI_CALL(ptiFlushAllViews());

  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

int XpuptiActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const pti_view_record_base*)> handler) {
  int count = 0;
  if (buf && validSize) {
    pti_view_record_base* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}

const std::pair<int, int> XpuptiActivityApi::processActivities(
    XpuptiActivityBufferMap& buffers,
    std::function<void(const pti_view_record_base*)> handler) {
  std::pair<int, int> res{0, 0};
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
  return res;
}

void XpuptiActivityApi::flushActivities() {
  XPUPTI_CALL(ptiFlushAllViews());
}

void XpuptiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
  XPUPTI_CALL(ptiFlushAllViews());
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_ = nullptr;
}

void XpuptiActivityApi::bufferCompletedTrampoline(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void XpuptiActivityApi::bufferCompleted(
    uint8_t* buffer,
    [[maybe_unused]] size_t size,
    size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedGpuTraceBuffers_.find(buffer);

  if (!readyGpuTraceBuffers_) {
    readyGpuTraceBuffers_ = std::make_unique<XpuptiActivityBufferMap>();
  }
  it->second->setSize(validSize);
  (*readyGpuTraceBuffers_)[it->first] = std::move(it->second);
  allocatedGpuTraceBuffers_.erase(it);
}

namespace {
void warnIfIttNotifyLibInvalid() noexcept {
  const auto itt_collector_path_env = std::getenv("INTEL_LIBITTNOTIFY64");
  if (itt_collector_path_env == nullptr) {
    LOG(WARNING) << "ENV variable `INTEL_LIBITTNOTIFY64` not set. "
                 << "XCCL host calls will not be collected.";
    return;
  }

  const std::filesystem::path itt_collector_path(itt_collector_path_env);

  if (itt_collector_path.is_relative()) {
    LOG(WARNING) << "`INTEL_LIBITTNOTIFY64` contains a relative path, "
                 << "an absolute path is recommended.";
  }

  std::error_code ec;
  if (not std::filesystem::exists(itt_collector_path, ec)) {
    LOG(WARNING)
        << "The library pointed to by `INTEL_LIBITTNOTIFY64` does not exist. "
        << "XCCL host calls will not be collected.";
  }
}
} // namespace

void XpuptiActivityApi::enableXpuptiActivities(
    const std::set<ActivityType>& selected_activities) {
  XPUPTI_CALL(ptiViewSetCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    switch (activity) {
      case ActivityType::GPU_MEMCPY:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
        break;

      case ActivityType::GPU_MEMSET:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
        break;

      case ActivityType::CONCURRENT_KERNEL:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL));
        break;

      case ActivityType::EXTERNAL_CORRELATION:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION));
        externalCorrelationEnabled_ = true;
        break;

      case ActivityType::XPU_RUNTIME:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_RUNTIME_API));
        XPUPTI_CALL(ptiViewEnableRuntimeApiClass(
            1, PTI_API_CLASS_GPU_OPERATION_CORE, PTI_API_GROUP_ALL));
        break;

      case ActivityType::XPU_DRIVER:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DRIVER_API));
        break;

      case ActivityType::XPU_SCOPE_PROFILER:
        // This case is handled in constructor of
        // XpuptiScopeProfilerSession
        break;

      case ActivityType::OVERHEAD:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_COLLECTION_OVERHEAD));
        break;

      case ActivityType::XPU_SYNC:
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_DEVICE_SYNCHRONIZATION));
        break;

      case ActivityType::COLLECTIVE_COMM:
        warnIfIttNotifyLibInvalid();
        XPUPTI_CALL(ptiViewEnable(PTI_VIEW_COMMUNICATION));
        break;

      default:
        break;
    }
  }
}

void XpuptiActivityApi::disablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
  for (const auto& activity : selected_activities) {
    switch (activity) {
      case ActivityType::GPU_MEMCPY:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
        break;

      case ActivityType::GPU_MEMSET:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
        break;

      case ActivityType::CONCURRENT_KERNEL:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL));
        break;

      case ActivityType::EXTERNAL_CORRELATION:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION));
        break;

      case ActivityType::XPU_RUNTIME:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_RUNTIME_API));
        break;

      case ActivityType::XPU_DRIVER:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DRIVER_API));
        break;

      case ActivityType::OVERHEAD:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_COLLECTION_OVERHEAD));
        break;

      case ActivityType::XPU_SYNC:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_DEVICE_SYNCHRONIZATION));
        break;

      case ActivityType::COLLECTIVE_COMM:
        XPUPTI_CALL(ptiViewDisable(PTI_VIEW_COMMUNICATION));
        break;

      default:
        break;
    }
  }
  externalCorrelationEnabled_ = false;
}

} // namespace KINETO_NAMESPACE
