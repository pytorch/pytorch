/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>
#include "CuptiActivity.h"
#include "CuptiActivityApi.h"
#include "GenericActivityProfiler.h"

namespace KINETO_NAMESPACE {

class CuptiActivityProfiler : public GenericActivityProfiler {
 public:
  CuptiActivityProfiler(CuptiActivityApi& cupti, bool cpuOnly);
  CuptiActivityProfiler(const CuptiActivityProfiler&) = delete;
  CuptiActivityProfiler& operator=(const CuptiActivityProfiler&) = delete;
  ~CuptiActivityProfiler() override = default;

 protected:
  void logGpuVersions() override;
  void setMaxGpuBufferSize(int64_t size) override;
  void enableGpuTracing() override;
  void disableGpuTracing() override;
  void clearGpuActivities() override;
  bool isGpuCollectionStopped() const override;
  void processGpuActivities(ActivityLogger& logger) override;
  void synchronizeGpuDevice() override;
  void pushCorrelationIdImpl(uint64_t id, CorrelationFlowType type) override;
  void popCorrelationIdImpl(CorrelationFlowType type) override;
  void onResetTraceData() override;
  void onFinalizeTrace(const Config& config, ActivityLogger& logger) override;

 private:
  void buildProcessingState(CuptiActivityBufferMap& buffers);
  // Process generic CUPTI activity
  void handleCuptiActivity(
      const CUpti_Activity* record,
      ActivityLogger* logger);
  // Process specific GPU activity types
  void handleCorrelationActivity(
      const CUpti_ActivityExternalCorrelation* correlation);
  void handleRuntimeActivity(
      const CUpti_ActivityAPI* activity,
      ActivityLogger* logger);
  void handleDriverActivity(
      const CUpti_ActivityAPI* activity,
      ActivityLogger* logger);
  void handleOverheadActivity(
      const CUpti_ActivityOverhead* activity,
      ActivityLogger* logger);
  void handleCudaEventActivity(
      const CUpti_ActivityCudaEventType* activity,
      ActivityLogger* logger);
  void handleCudaSyncActivity(
      const CUpti_ActivitySynchronization* activity,
      ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);
  void logDeferredEvents();

  // Calls to CUPTI is encapsulated behind this interface
  CuptiActivityApi& cupti_;
};

// Helper function to map context ID to device ID
uint32_t contextIdtoDeviceId(uint32_t contextId);

} // namespace KINETO_NAMESPACE
