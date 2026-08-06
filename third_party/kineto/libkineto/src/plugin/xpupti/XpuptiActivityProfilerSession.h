/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiProfilerMacros.h"

#include "IActivityProfiler.h"
#include "libkineto.h"

#include <pti/pti_view.h>

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace KINETO_NAMESPACE {

class XpuptiActivityApi;

using DeviceUUIDsT = std::array<unsigned char, 16>;

class XpuptiActivityProfilerSession
    : public libkineto::IActivityProfilerSession {
 public:
  XpuptiActivityProfilerSession() = delete;
  XpuptiActivityProfilerSession(
      XpuptiActivityApi& xpti,
      const std::string& name,
      const libkineto::Config& config,
      const std::set<ActivityType>& activity_types);
  XpuptiActivityProfilerSession(const XpuptiActivityProfilerSession&) = delete;
  XpuptiActivityProfilerSession& operator=(
      const XpuptiActivityProfilerSession&) = delete;

  ~XpuptiActivityProfilerSession();

  void start() override;
  void stop() override;
  void toggleCollectionDynamic(const bool enable) override;
  std::vector<std::string> errors() override {
    return errors_;
  };
  void processTrace(ActivityLogger& logger) override;
  void processTrace(
      ActivityLogger& logger,
      libkineto::getLinkedActivityCallback get_linked_activity,
      int64_t captureWindowStartTime,
      int64_t captureWindowEndTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return {};
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void pushUserCorrelationId(uint64_t id) override;
  void popUserCorrelationId() override;

  // Whether a runtime/driver record starts a CPU->GPU flow arrow. Only host
  // runtime (XPU_RUNTIME) records do; driver (XPU_DRIVER) records share the
  // same correlation id and would otherwise create a duplicate flow start.
  // Static so it can be unit-tested without real hardware.
  static bool startsFlow(ActivityType activityType);

 private:
  void checkTimestampOrder(const ITraceActivity* act1);
  void removeCorrelatedPtiActivities(const ITraceActivity* act1);
  bool outOfRange(const ITraceActivity* act);
  int64_t getMappedQueueId(uint64_t sycl_queue_id);
  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);
  void handleCorrelationActivity(
      const pti_view_record_external_correlation* correlation);

  using pti_view_record_api_t = pti_view_record_api;

  template <typename PTI_VIEW>
  std::string getApiName(const PTI_VIEW* activity) {
    const char* api_name = nullptr;
    XPUPTI_CALL(ptiViewGetApiIdName(
        activity->_api_group, activity->_api_id, &api_name));
    return std::string(api_name);
  }

  template <class pti_view_memory_record_type>
  void handleRuntimeKernelMemcpyMemsetActivities(
      ActivityType activityType,
      const pti_view_memory_record_type* activity,
      ActivityLogger& logger);

  void handleSynchronizationActivity(
      const pti_view_record_synchronization* activity,
      ActivityLogger& logger);
  void handleCommunicationActivity(
      const pti_view_record_comms* activity,
      ActivityLogger& logger);
  void handleOverheadActivity(
      const pti_view_record_overhead* activity,
      ActivityLogger& logger);
  void handlePtiActivity(
      const pti_view_record_base* record,
      ActivityLogger& logger);

  // enumerate XPU Device UUIDs from runtime for once
  void enumDeviceUUIDs();

  // get logical device index(int8) from the given UUID from runtime
  // for profiling activity creation
  DeviceIndex_t getDeviceIdxFromUUID(const uint8_t deviceUUID[16]);

  void addResouceInfo(int32_t device_id, int32_t sycl_queue_id);

 protected:
  static uint32_t iterationCount_;
  static std::vector<DeviceUUIDsT> deviceUUIDs_;

  int64_t captureWindowStartTime_{0};
  int64_t captureWindowEndTime_{0};
  int64_t profilerStartTs_{0};
  int64_t profilerEndTs_{0};
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;
  std::unordered_map<int64_t, const ITraceActivity*> correlatedPtiActivities_;
  std::map<
      std::tuple<int32_t, int32_t, int64_t>,
      libkineto::GenericTraceActivity*>
      userAnnotationsByStream_;
  std::vector<std::string> errors_;

  libkineto::getLinkedActivityCallback cpuActivity_;

  XpuptiActivityApi& xpti_;
  libkineto::CpuTraceBuffer traceBuffer_;
  std::vector<std::pair<int32_t, int32_t>> resourceInfo_;
  std::unique_ptr<const libkineto::Config> config_{nullptr};
  const std::set<ActivityType>& activity_types_;
  std::string name_;

  struct KernelActivity {
    void emplace(
        int64_t startTime,
        int64_t endTime,
        int32_t device,
        int32_t resource) {
      startTime_ = startTime;
      endTime_ = endTime;
      device_ = device;
      resource_ = resource;
    }

    int64_t startTime_{0};
    int64_t endTime_{0};
    int32_t device_{0};
    int32_t resource_{0};
  };

  std::unordered_map<uint64_t, KernelActivity> kernelActivities_;
  uint64_t lastKernelActivityEndTime_{0};
};

} // namespace KINETO_NAMESPACE
