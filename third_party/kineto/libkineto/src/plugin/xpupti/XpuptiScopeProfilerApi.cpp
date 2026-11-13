/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiScopeProfilerApi.h"
#include "Config.h"
#include "ThrowUtil.h"
#include "XpuptiProfilerMacros.h"
#include "XpuptiScopeProfilerConfig.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <pti/pti.h>
#include <pti/pti_metrics.h>

namespace KINETO_NAMESPACE {

std::vector<pti_device_handle_t> selectDeviceHandles(
    std::span<const pti_device_handle_t> handles,
    std::span<const int> indices) {
  const auto outOfRange = [handles](int idx) {
    return idx < 0 || std::cmp_greater_equal(idx, handles.size());
  };
  if (const auto bad = std::ranges::find_if(indices, outOfRange);
      bad != indices.end()) {
    KINETO_THROW(
        std::runtime_error,
        fmt::format(
            "XPUPTI_PROFILER_DEVICES index {} is out of range; {} XPU device(s) available",
            *bad,
            handles.size()));
  }
  // Gather: map each requested index to its device handle, preserving order.
  std::vector<pti_device_handle_t> selected(indices.size());
  std::ranges::transform(indices, selected.begin(), [handles](int idx) {
    return handles[static_cast<std::size_t>(idx)];
  });
  return selected;
}

XpuptiScopeProfilerApi::safe_pti_scope_collection_handle_t::
    safe_pti_scope_collection_handle_t(std::exception_ptr& exceptFromDestructor)
    : exceptFromDestructor_(exceptFromDestructor) {
  XPUPTI_CALL(ptiMetricsScopeEnable(&handle_));
}

XpuptiScopeProfilerApi::safe_pti_scope_collection_handle_t::
    ~safe_pti_scope_collection_handle_t() noexcept {
  try {
    XPUPTI_CALL(ptiMetricsScopeDisable(handle_));
  } catch (...) {
    exceptFromDestructor_ = std::current_exception();
  }
}

void XpuptiScopeProfilerApi::enableScopeProfiler(const Config& cfg) {
  uint32_t deviceCount = 0;
  XPUPTI_CALL(ptiMetricsGetDevices(nullptr, &deviceCount));

  if (deviceCount == 0) {
    KINETO_THROW(std::runtime_error, "No XPU devices available");
  }

  auto devices = std::make_unique<pti_device_properties_t[]>(deviceCount);
  XPUPTI_CALL(ptiMetricsGetDevices(devices.get(), &deviceCount));

  auto devicesHandles = std::make_unique<pti_device_handle_t[]>(deviceCount);
  for (uint32_t i = 0; i < deviceCount; ++i) {
    devicesHandles[i] = devices[i]._handle;
  }

  const auto& spcfg = XpuptiScopeProfilerConfig::get(cfg);
  const auto& activitiesXpuptiMetrics = spcfg.activitiesXpuptiMetrics();

  std::vector<const char*> metricNames;
  metricNames.reserve(activitiesXpuptiMetrics.size());
  std::transform(
      activitiesXpuptiMetrics.begin(),
      activitiesXpuptiMetrics.end(),
      std::back_inserter(metricNames),
      [](const std::string& s) { return s.c_str(); });

  pti_metrics_scope_mode_t collectionMode = spcfg.xpuptiProfilerPerKernel()
      ? PTI_METRICS_SCOPE_AUTO_KERNEL
      : PTI_METRICS_SCOPE_USER;

  if (collectionMode == PTI_METRICS_SCOPE_USER) {
    KINETO_THROW(
        std::runtime_error,
        "XPUPTI_PROFILER_ENABLE_PER_KERNEL has to be set to 1. Other variants are currently not supported.");
  }

  scopeHandleOpt_.emplace(exceptFromScopeHandleDestructor_);

  const auto& requestedDevices = spcfg.xpuptiProfilerDevices();

#if PTI_VERSION_AT_LEAST(0, 18)
  if (requestedDevices.empty()) {
    // Auto-detect: PTI profiles whichever devices the workload actually uses.
    XPUPTI_CALL(ptiMetricsScopeConfigure(
        *scopeHandleOpt_,
        collectionMode,
        /*devices_to_profile=*/nullptr,
        /*device_count=*/0,
        metricNames.data(),
        metricNames.size()));
  } else {
    // Explicit subset: map requested indices to device handles.
    auto selectedHandles = selectDeviceHandles(
        {devicesHandles.get(), deviceCount}, requestedDevices);
    XPUPTI_CALL(ptiMetricsScopeConfigure(
        *scopeHandleOpt_,
        collectionMode,
        selectedHandles.data(),
        static_cast<uint32_t>(selectedHandles.size()),
        metricNames.data(),
        metricNames.size()));
  }
#else
  // PTI < 0.18 (pre PTI-363): multi-device metrics scope is not available;
  // ptiMetricsScopeConfigure accepts only a single device.
  if (requestedDevices.size() > 1) {
    KINETO_THROW(
        std::runtime_error,
        "XPUPTI_PROFILER_DEVICES lists more than one device, but this build "
        "links PTI < 0.18 which supports only single-device metrics scope. "
        "Rebuild against PTI >= 0.18 for multi-device support.");
  }
  // Point at the single requested device (default: first device).
  pti_device_handle_t singleHandle = requestedDevices.empty()
      ? devicesHandles[0]
      : selectDeviceHandles(
            {devicesHandles.get(), deviceCount}, requestedDevices)
            .front();
  XPUPTI_CALL(ptiMetricsScopeConfigure(
      *scopeHandleOpt_,
      collectionMode,
      &singleHandle,
      1,
      metricNames.data(),
      metricNames.size()));
#endif

  uint64_t expectedKernels = spcfg.xpuptiProfilerMaxScopes();
  size_t estimatedCollectionBufferSize = 0;
  XPUPTI_CALL(ptiMetricsScopeQueryCollectionBufferSize(
      *scopeHandleOpt_, expectedKernels, &estimatedCollectionBufferSize));

  XPUPTI_CALL(ptiMetricsScopeSetCollectionBufferSize(
      *scopeHandleOpt_, estimatedCollectionBufferSize));
}

void XpuptiScopeProfilerApi::disableScopeProfiler() {
  scopeHandleOpt_.reset();
  if (exceptFromScopeHandleDestructor_) {
    std::rethrow_exception(exceptFromScopeHandleDestructor_);
  }
}

void XpuptiScopeProfilerApi::startScopeActivity() {
  if (scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeStartCollection(*scopeHandleOpt_));
  }
}

void XpuptiScopeProfilerApi::stopScopeActivity() {
  if (scopeHandleOpt_) {
    XPUPTI_CALL(ptiMetricsScopeStopCollection(*scopeHandleOpt_));
  }
}

static size_t IntDivRoundUp(size_t a, size_t b) {
  return (a + b - 1) / b;
}

void XpuptiScopeProfilerApi::processScopeTrace(
    const std::function<void(
        const pti_metrics_scope_record_t*,
        const pti_metrics_scope_record_metadata_t& metadata)>& handler) {
  if (scopeHandleOpt_) {
    pti_metrics_scope_record_metadata_t metadata;
    metadata._struct_size = sizeof(pti_metrics_scope_record_metadata_t);

    XPUPTI_CALL(ptiMetricsScopeGetMetricsMetadata(*scopeHandleOpt_, &metadata));

    uint64_t collectionBuffersCount = 0;
    XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffersCount(
        *scopeHandleOpt_, &collectionBuffersCount));

    for (uint64_t bufferId = 0; bufferId < collectionBuffersCount; ++bufferId) {
      void* collectionBuffer = nullptr;
      size_t actualCollectionBufferSize = 0;
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBuffer(
          *scopeHandleOpt_,
          bufferId,
          &collectionBuffer,
          &actualCollectionBufferSize));

      pti_metrics_scope_collection_buffer_properties_t metricsBufferProps;
      metricsBufferProps._struct_size =
          sizeof(pti_metrics_scope_collection_buffer_properties_t);
      XPUPTI_CALL(ptiMetricsScopeGetCollectionBufferProperties(
          *scopeHandleOpt_, collectionBuffer, &metricsBufferProps));

      size_t requiredMetricsBufferSize = 0;
      size_t recordsCount = 0;
      XPUPTI_CALL(ptiMetricsScopeQueryMetricsBufferSize(
          *scopeHandleOpt_,
          collectionBuffer,
          &requiredMetricsBufferSize,
          &recordsCount));

      if (recordsCount > 0) {
        auto metricsBuffer =
            std::make_unique<pti_metrics_scope_record_t[]>(IntDivRoundUp(
                requiredMetricsBufferSize, sizeof(pti_metrics_scope_record_t)));

        size_t actualRecordsCount = 0;
        XPUPTI_CALL(ptiMetricsScopeCalculateMetrics(
            *scopeHandleOpt_,
            collectionBuffer,
            metricsBuffer.get(),
            requiredMetricsBufferSize,
            &actualRecordsCount));

        for (size_t recordId = 0; recordId < actualRecordsCount; ++recordId) {
          auto record = metricsBuffer.get() + recordId;
          handler(record, metadata);
        }
      }
    }
  }
}

} // namespace KINETO_NAMESPACE
