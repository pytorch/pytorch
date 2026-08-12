/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <set>

#include <cupti.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityType.h"
#include "CuptiActivityBuffer.h"
#include "CuptiCallbackApi.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

class CuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };
  // Control Variables shared with CuptiCallbackApi for teardown
  std::atomic<uint32_t> teardownCupti_{0};
  std::mutex finalizeMutex_;
  std::condition_variable finalizeCond_;

  CuptiActivityApi() = default;
  CuptiActivityApi(const CuptiActivityApi&) = delete;
  CuptiActivityApi& operator=(const CuptiActivityApi&) = delete;

  virtual ~CuptiActivityApi() = default;

  static CuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCuptiActivities(
      const std::set<ActivityType>& selected_activities,
      bool enablePerThreadBuffers = false);
  void disableCuptiActivities(
      const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void flushActivities();
  void teardownContext();

  virtual std::unique_ptr<CuptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, size_t> processActivities(
      CuptiActivityBufferMap&,
      const std::function<void(const CUpti_Activity*)>& handler);

  void setMaxBufferSize(int64_t size);
  void setDeviceBufferSize(size_t size);
  void setDeviceBufferPoolLimit(size_t limit);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

  static void forceLoadCupti();

  // CUPTI configuraiton that needs to be set before CUDA context creation
  static void preConfigureCUPTI();

 private:
  int64_t maxGpuBufferCount_{0};
  CuptiActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<CuptiActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  std::atomic<uint32_t> tearingDown_{0};
  std::atomic<bool> externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      const std::function<void(const CUpti_Activity*)>& handler);
  static void CUPTIAPI bufferRequestedTrampoline(
      uint8_t** buffer,
      size_t* size,
      size_t* maxNumRecords);
  static void CUPTIAPI bufferCompletedTrampoline(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(
      CUcontext ctx,
      uint32_t streamId,
      uint8_t* buffer,
      size_t /* unused */,
      size_t validSize);
};

} // namespace KINETO_NAMESPACE
