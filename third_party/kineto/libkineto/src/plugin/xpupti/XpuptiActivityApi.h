/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiActivityBuffer.h"
#include "XpuptiProfilerMacros.h"

#include "ActivityType.h"

#include <pti/pti_view.h>

#include <functional>
#include <mutex>

namespace KINETO_NAMESPACE {

class XpuptiActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  XpuptiActivityApi() = default;
  XpuptiActivityApi(const XpuptiActivityApi&) = delete;
  XpuptiActivityApi& operator=(const XpuptiActivityApi&) = delete;

  virtual ~XpuptiActivityApi() {}

  static XpuptiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableXpuptiActivities(
      const std::set<ActivityType>& selected_activities);
  void disablePtiActivities(const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void flushActivities();

  virtual std::unique_ptr<XpuptiActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      XpuptiActivityBufferMap&,
      std::function<void(const pti_view_record_base*)> handler);

 private:
  XpuptiActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XpuptiActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const pti_view_record_base*)> handler);
  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size);
  static void bufferCompletedTrampoline(
      uint8_t* buffer,
      size_t size,
      size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

} // namespace KINETO_NAMESPACE
