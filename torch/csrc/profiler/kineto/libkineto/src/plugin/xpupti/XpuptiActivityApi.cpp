/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityApi.h"
#include "ILoggerObserver.h"
#include "Logger.h"
#include "XpuptiActivityBuffer.h"
#include "XpuptiProfilerMacros.h"

#include <cstdlib>
#include <filesystem>
#include <map>
#include <ostream>
#include <span>
#include <system_error>

#include <pti/pti.h>

namespace KINETO_NAMESPACE {

constexpr std::size_t kBufSize(4 * 1024 * 1024);

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
    std::uint8_t* buffer,
    std::size_t valid_size,
    pti_view_record_base*& record) {
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
  return record != nullptr;
}

void XpuptiActivityApi::bufferRequestedTrampoline(
    std::uint8_t** buffer,
    std::size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XpuptiActivityApi::bufferRequested(
    std::uint8_t** buffer,
    std::size_t* size) {
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
      if (readyGpuTraceBuffers_) {
        return std::move(readyGpuTraceBuffers_);
      }
      return nullptr;
    }
  }

  flushActivities();

  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

std::size_t XpuptiActivityApi::processActivitiesForBuffer(
    std::uint8_t* buf,
    std::size_t validSize,
    const std::function<void(const pti_view_record_base*)>& handler) {
  std::size_t count = 0;
  if (buf && validSize) {
    pti_view_record_base* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}

ActivitiesStats XpuptiActivityApi::processActivities(
    XpuptiActivityBufferMap& buffers,
    const std::function<void(const pti_view_record_base*)>& handler) {
  ActivitiesStats res{};
  for (const auto& [_, buf] : buffers) {
    res.activitiesCount +=
        processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.buffersSize += buf->size();
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
      readyGpuTraceBuffers_.reset();
      return;
    }
  }
  flushActivities();
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_.reset();
}

void XpuptiActivityApi::bufferCompletedTrampoline(
    std::uint8_t* buffer,
    std::size_t size,
    std::size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void XpuptiActivityApi::bufferCompleted(
    std::uint8_t* buffer,
    [[maybe_unused]] std::size_t size,
    std::size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);

  auto it = allocatedGpuTraceBuffers_.find(buffer);
  if (it == std::end(allocatedGpuTraceBuffers_)) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << static_cast<void*>(buffer);
    return;
  }

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

template <pti_view_kind... kViews>
constexpr pti_view_kind kViewKinds[] = {kViews...};

// The PTI views an activity type is collected through, if it has any of its
// own. Shared by enable and disable so the two cannot drift apart.
constexpr std::span<const pti_view_kind> findPtiViewKinds(
    ActivityType activity) {
  switch (activity) {
    case ActivityType::GPU_MEMCPY:
      // PTI reports a peer-to-peer copy only through its own view.
      return kViewKinds<
          PTI_VIEW_DEVICE_GPU_MEM_COPY,
          PTI_VIEW_DEVICE_GPU_MEM_COPY_P2P>;
    case ActivityType::GPU_MEMSET:
      return kViewKinds<PTI_VIEW_DEVICE_GPU_MEM_FILL>;
    case ActivityType::CONCURRENT_KERNEL:
      return kViewKinds<PTI_VIEW_DEVICE_GPU_KERNEL>;
    case ActivityType::EXTERNAL_CORRELATION:
      return kViewKinds<PTI_VIEW_EXTERNAL_CORRELATION>;
    case ActivityType::XPU_RUNTIME:
      return kViewKinds<PTI_VIEW_RUNTIME_API>;
    case ActivityType::XPU_DRIVER:
      return kViewKinds<PTI_VIEW_DRIVER_API>;
    case ActivityType::OVERHEAD:
      return kViewKinds<PTI_VIEW_COLLECTION_OVERHEAD>;
    case ActivityType::XPU_SYNC:
      return kViewKinds<PTI_VIEW_DEVICE_SYNCHRONIZATION>;
    case ActivityType::COLLECTIVE_COMM:
      return kViewKinds<PTI_VIEW_COMMUNICATION>;
    default:
      // XPU_SCOPE_PROFILER among others: enabled by
      // XpuptiScopeProfilerSession's constructor, not by a view.
      return {};
  }
}

void enableDriverMemoryApis() {
  static constexpr pti_api_id_driver_levelzero kAllowedMemoryApis[]{
      pti_api_id_driver_levelzero::zeMemAllocDevice_id,
      pti_api_id_driver_levelzero::zeMemAllocHost_id,
      pti_api_id_driver_levelzero::zeMemAllocShared_id,
      pti_api_id_driver_levelzero::zeMemCloseIpcHandle_id,
      pti_api_id_driver_levelzero::zeMemFreeExt_id,
      pti_api_id_driver_levelzero::zeMemFree_id,
      pti_api_id_driver_levelzero::zeMemGetFileDescriptorFromIpcHandleExp_id,
      pti_api_id_driver_levelzero::zeMemGetIpcHandleFromFileDescriptorExp_id,
      pti_api_id_driver_levelzero::zeMemGetIpcHandleWithProperties_id,
      pti_api_id_driver_levelzero::zeMemGetIpcHandle_id,
      pti_api_id_driver_levelzero::zeMemOpenIpcHandle_id,
      pti_api_id_driver_levelzero::zeMemPutIpcHandle_id,
      pti_api_id_driver_levelzero::zePhysicalMemCreate_id,
      pti_api_id_driver_levelzero::zePhysicalMemDestroy_id,
      pti_api_id_driver_levelzero::zeVirtualMemFree_id,
      pti_api_id_driver_levelzero::zeVirtualMemMap_id,
      pti_api_id_driver_levelzero::zeVirtualMemReserve_id,
      pti_api_id_driver_levelzero::zeVirtualMemUnmap_id,
  };

  // No class scopes allocation, so these go in one id at a time.
  for (const auto memoryApi : kAllowedMemoryApis)
    XPUPTI_CALL(ptiViewEnableDriverApi(
        /*enable=*/1, PTI_API_GROUP_LEVELZERO, memoryApi));
}
} // namespace

void XpuptiActivityApi::enableXpuptiActivities(
    const ActivityTypeMask& selected_activities) {
  XPUPTI_CALL(ptiViewSetCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  selected_activities.forEach([this](ActivityType activity) {
    const auto views = findPtiViewKinds(activity);
    if (views.empty()) {
      return;
    }
    for (const auto view : views) {
      XPUPTI_CALL(ptiViewEnable(view));
    }

    // What enabling a view takes on top of the view itself.
    switch (activity) {
      case ActivityType::XPU_RUNTIME:
        XPUPTI_CALL(ptiViewEnableRuntimeApiClass(
            /*enable=*/1, PTI_API_CLASS_GPU_OPERATION_CORE, PTI_API_GROUP_ALL));
        break;

      case ActivityType::XPU_DRIVER:
        // ptiViewEnable above left every L0 id enabled.
        // The class call below is the first to set granularity,
        // which resets the group to all-disabled before applying,
        // so the per-id calls after it only add.
        XPUPTI_CALL(ptiViewEnableDriverApiClass(
            /*enable=*/1,
            PTI_API_CLASS_GPU_OPERATION_CORE,
            PTI_API_GROUP_LEVELZERO));
        enableDriverMemoryApis();
        break;

      case ActivityType::EXTERNAL_CORRELATION:
        externalCorrelationEnabled_ = true;
        break;

      case ActivityType::COLLECTIVE_COMM:
        warnIfIttNotifyLibInvalid();
        break;

      default:
        break;
    }
  });
}

void XpuptiActivityApi::disablePtiActivities(
    const ActivityTypeMask& selected_activities) {
  selected_activities.forEach([](ActivityType activity) {
    for (const auto view : findPtiViewKinds(activity)) {
      XPUPTI_CALL(ptiViewDisable(view));
    }
  });
  externalCorrelationEnabled_ = false;
}

} // namespace KINETO_NAMESPACE
