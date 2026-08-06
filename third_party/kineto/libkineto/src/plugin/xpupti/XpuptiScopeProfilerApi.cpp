/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "XpuptiScopeProfilerApi.h"
#include "XpuptiScopeProfilerConfig.h"

namespace KINETO_NAMESPACE {

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
    throw std::runtime_error("No XPU devices available");
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
    throw std::runtime_error(
        "XPUPTI_PROFILER_ENABLE_PER_KERNEL has to be set to 1. Other variants are currently not supported.");
  }

  scopeHandleOpt_.emplace(exceptFromScopeHandleDestructor_);
  XPUPTI_CALL(ptiMetricsScopeConfigure(
      *scopeHandleOpt_,
      collectionMode,
      devicesHandles.get(),
      ((void)deviceCount, 1), // Only 1 device is currently supported
      metricNames.data(),
      metricNames.size()));

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
    std::function<void(
        const pti_metrics_scope_record_t*,
        const pti_metrics_scope_record_metadata_t& metadata)> handler) {
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
