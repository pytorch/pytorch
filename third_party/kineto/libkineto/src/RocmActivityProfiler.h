/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO DEVICE_AGNOSTIC: The build system should be strict enough that we don't
//                       need this guard in the header file.
#ifdef HAS_ROCTRACER

#include <cstdint>

#include <rocprofiler-sdk/version.h>

#include "GenericActivityProfiler.h"
#include "RocLogger.h"
#include "RocprofActivityApi.h"

namespace KINETO_NAMESPACE {

class RocmActivityProfiler : public GenericActivityProfiler {
 public:
  RocmActivityProfiler(RocprofActivityApi& rocprof, bool cpuOnly);
  RocmActivityProfiler(const RocmActivityProfiler&) = delete;
  RocmActivityProfiler& operator=(const RocmActivityProfiler&) = delete;
  ~RocmActivityProfiler() override = default;

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
  // Process generic RocProf activity
  void handleRocprofActivity(const rocprofBase* record, ActivityLogger* logger);
  void handleCorrelationActivity(
      uint64_t correlationId,
      uint64_t externalId,
      RocLogger::CorrelationDomain externalKind);
  // Process specific GPU activity types
  template <class T>
  void handleRuntimeActivity(const T* activity, ActivityLogger* logger);
  void handleGpuActivity(const rocprofAsyncRow* record, ActivityLogger* logger);

  // Calls to rocprofiler-sdk is encapsulated behind this interface
  RocprofActivityApi& roc_;
};

} // namespace KINETO_NAMESPACE

#endif // HAS_ROCTRACER
