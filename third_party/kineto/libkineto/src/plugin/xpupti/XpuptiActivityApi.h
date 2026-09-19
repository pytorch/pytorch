/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"
#include "XpuptiActivityBuffer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

#include <pti/pti_view.h>

namespace KINETO_NAMESPACE {

struct ActivitiesStats {
  std::size_t activitiesCount;
  std::size_t buffersSize; // Valid bytes
};

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

  virtual ActivitiesStats processActivities(
      XpuptiActivityBufferMap&,
      const std::function<void(const pti_view_record_base*)>& handler);

 private:
  XpuptiActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XpuptiActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};

  std::size_t processActivitiesForBuffer(
      std::uint8_t* buf,
      std::size_t validSize,
      const std::function<void(const pti_view_record_base*)>& handler);
  static void bufferRequestedTrampoline(
      std::uint8_t** buffer,
      std::size_t* size);
  static void bufferCompletedTrampoline(
      std::uint8_t* buffer,
      std::size_t size,
      std::size_t validSize);

 protected:
  void bufferRequested(std::uint8_t** buffer, std::size_t* size);
  void bufferCompleted(
      std::uint8_t* buffer,
      std::size_t size,
      std::size_t validSize);
};

} // namespace KINETO_NAMESPACE
