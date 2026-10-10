/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

#include <fmt/format.h>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiCbidRegistry.h"
#include "CuptiTimestamp.h"
#include "Demangle.h"
#include "DeviceProperties.h"
#include "GenericTraceActivity.h"
#include "ITraceActivity.h"
#include "Logger.h"
#include "MetadataFieldCatalog.h"
#include "ThreadUtil.h"
#include "TypedMetadataJson.h"
#include "cupti_strings.h"
#include "output_base.h"

namespace libkineto {
class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

// These classes wrap the various CUPTI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Cupti activity type
template <class T>
struct CuptiActivity : public ITraceActivity {
  explicit CuptiActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  int64_t timestamp() const override {
    return convertCuptiTimestamp(activity_.start);
  }

  int64_t duration() const override {
    return convertCuptiDuration(activity_.start, activity_.end);
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override {
    return 0;
  }
  int32_t getThreadId() const override {
    return 0;
  }
  const ITraceActivity* linkedActivity() const override {
    return linked_;
  }
  int flowType() const override {
    return kLinkAsyncCpuGpu;
  }
  int64_t flowId() const override {
    return correlationId();
  }
  const T& raw() const {
    return activity_;
  }
  const TraceSpan* traceSpan() const override {
    return nullptr;
  }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// CUpti_ActivityAPI - CUDA runtime activities
struct RuntimeActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit RuntimeActivity(
      const CUpti_ActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_.correlationId;
  }
  int64_t deviceId() const override {
    return processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::CUDA_RUNTIME;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return runtimeCbidName(activity_.cbid);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA driver activities
struct DriverActivity : public CuptiActivity<CUpti_ActivityAPI> {
  explicit DriverActivity(
      const CUpti_ActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CuptiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_.correlationId;
  }
  int64_t deviceId() const override {
    return processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::CUDA_DRIVER;
  }
  bool flowStart() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivityAPI - CUDA runtime activities
struct OverheadActivity : public CuptiActivity<CUpti_ActivityOverhead> {
  explicit OverheadActivity(
      const CUpti_ActivityOverhead* activity,
      const ITraceActivity* linked,
      int32_t threadId = 0)
      : CuptiActivity(activity, linked), threadId_(threadId) {}

  int64_t timestamp() const override {
    return convertCuptiTimestamp(activity_.start);
  }

  int64_t duration() const override {
    return convertCuptiDuration(activity_.start, activity_.end);
  }

  // TODO: Update this with PID ordering
  int64_t deviceId() const override {
    return -1;
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::OVERHEAD;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return overheadKindString(activity_.overheadKind);
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;

 private:
  const int32_t threadId_;
};

// CUpti_ActivitySynchronization - CUDA synchronization events
struct CudaSyncActivity : public CuptiActivity<CUpti_ActivitySynchronization> {
  explicit CudaSyncActivity(
      const CUpti_ActivitySynchronization* activity,
      const ITraceActivity* linked,
      int32_t srcStream,
      int32_t srcCorrId)
      : CuptiActivity(activity, linked),
        srcStream_(srcStream),
        srcCorrId_(srcCorrId) {}
  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override;
  int64_t resourceId() const override;
  ActivityType type() const override {
    return ActivityType::CUDA_SYNC;
  }
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;
  const CUpti_ActivitySynchronization& raw() const {
    return CuptiActivity<CUpti_ActivitySynchronization>::raw();
  }

 private:
  const int32_t srcStream_;
  const int32_t srcCorrId_;
};

// Use CUpti_ActivityCudaEvent2 in CUDA 12.8+ for enhanced event tracking
#if CUDA_VERSION >= 12080
using CUpti_ActivityCudaEventType = CUpti_ActivityCudaEvent2;
#else
using CUpti_ActivityCudaEventType = CUpti_ActivityCudaEvent;
#endif

// Template specialization for timestamp() and duration() for CudaEvent types
template <>
inline int64_t CuptiActivity<CUpti_ActivityCudaEventType>::timestamp() const {
#if CUDA_VERSION >= 12080
  return convertCuptiTimestamp(activity_.deviceTimestamp);
#else
  // For CUDA < 12.8, deviceTimestamp doesn't exist, set to 0
  return 0;
#endif
}

template <>
inline int64_t CuptiActivity<CUpti_ActivityCudaEventType>::duration() const {
  // Set duration to 1 to make events visible in trace
  return 1;
}

// CUpti_ActivityCudaEvent - CUDA event activities
struct CudaEventActivity : public CuptiActivity<CUpti_ActivityCudaEventType> {
  explicit CudaEventActivity(
      const CUpti_ActivityCudaEventType* activity,
      const ITraceActivity* linked)
      : CuptiActivity(activity, linked) {}

  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override;
  int64_t resourceId() const override;
  ActivityType type() const override {
    return ActivityType::CUDA_EVENT;
  }
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;
  const CUpti_ActivityCudaEventType& raw() const {
    return CuptiActivity<CUpti_ActivityCudaEventType>::raw();
  }
};

// Base class for GPU activities.
// Can also be instantiated directly.
template <class T>
struct GpuActivity : public CuptiActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity* linked)
      : CuptiActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override {
    return raw().deviceId;
  }
  int64_t resourceId() const override {
    return raw().streamId;
  }
  ActivityType type() const override;
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override;
  const T& raw() const {
    return CuptiActivity<T>::raw();
  }
};

// forward declaration
uint32_t contextIdtoDeviceId(uint32_t contextId);

template <>
inline const std::string GpuActivity<CUpti_ActivityKernelType>::name() const {
  return demangle(raw().name);
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityKernelType>::type() const {
  return ActivityType::CONCURRENT_KERNEL;
}

inline bool isWaitEventSync(CUpti_ActivitySynchronizationType type) {
  return (type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline bool isEventSync(CUpti_ActivitySynchronizationType type) {
  return (
      type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE ||
      type == CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT);
}

inline int32_t streamIdForTrace(uint32_t streamId) {
  // CUPTI uses all bits set when no stream applies. Preserve the legacy trace
  // representation where that sentinel appears as -1 instead of 4294967295.
  if (streamId == static_cast<uint32_t>(CUPTI_SYNCHRONIZATION_INVALID_VALUE)) {
    return -1;
  }
  return static_cast<int32_t>(streamId);
}

inline const std::string CudaSyncActivity::name() const {
  return syncTypeString(raw().type);
}

inline int64_t CudaSyncActivity::deviceId() const {
  return contextIdtoDeviceId(raw().contextId);
}

inline int64_t CudaSyncActivity::resourceId() const {
  return streamIdForTrace(raw().streamId);
}

inline void CudaSyncActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline const std::string CudaSyncActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void CudaSyncActivity::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivitySynchronization& sync = raw();
  visitor.visit(
      CudaMetadataFields::kCudaSyncKind,
      std::string{syncTypeString(sync.type)});
  if (isEventSync(sync.type)) {
    visitor.visit(
        CudaMetadataFields::kWaitOnStream, static_cast<int64_t>(srcStream_));
    visitor.visit(
        CudaMetadataFields::kWaitOnCudaEventRecordCorrId,
        static_cast<int64_t>(srcCorrId_));
    visitor.visit(
        CudaMetadataFields::kWaitOnCudaEventId,
        static_cast<uint64_t>(sync.cudaEventId));
  }
  visitor.visit(
      CudaMetadataFields::kStream,
      static_cast<int64_t>(streamIdForTrace(sync.streamId)));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(sync.correlationId));
  visitor.visit(CudaMetadataFields::kDevice, deviceId());
  visitor.visit(
      CudaMetadataFields::kContext, static_cast<uint64_t>(sync.contextId));
}

inline const std::string CudaEventActivity::name() const {
  return "cudaEvent";
}

inline int64_t CudaEventActivity::deviceId() const {
  return contextIdtoDeviceId(raw().contextId);
}

inline int64_t CudaEventActivity::resourceId() const {
  return streamIdForTrace(raw().streamId);
}

inline void CudaEventActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline const std::string CudaEventActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void CudaEventActivity::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivityCudaEventType& event = raw();
  visitor.visit(
      CudaMetadataFields::kEventId, static_cast<uint64_t>(event.eventId));
  visitor.visit(
      CudaMetadataFields::kStream,
      static_cast<int64_t>(streamIdForTrace(event.streamId)));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(event.correlationId));
  visitor.visit(CudaMetadataFields::kDevice, deviceId());
  visitor.visit(
      CudaMetadataFields::kContext, static_cast<uint64_t>(event.contextId));
}

template <class T>
inline void GpuActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

constexpr int64_t us(int64_t timestamp) {
  // It's important that this conversion is the same here and in the CPU trace.
  // No rounding!
  return timestamp / 1000;
}

template <class T>
inline void addGraphNodeTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor,
    [[maybe_unused]] const T& activity) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  visitor.visit(
      CudaMetadataFields::kGraphId, static_cast<uint64_t>(activity.graphId));
  visitor.visit(
      CudaMetadataFields::kGraphNodeId,
      static_cast<uint64_t>(activity.graphNodeId));
#endif
}

template <class T>
inline void addPriorityTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor,
    [[maybe_unused]] const T& kernel) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010
  visitor.visit(
      CudaMetadataFields::kPriority, static_cast<int64_t>(kernel.priority));
#endif
}

template <class T>
inline void addChannelTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor,
    [[maybe_unused]] const T& activity) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  visitor.visit(
      CudaMetadataFields::kChannel, static_cast<uint64_t>(activity.channelID));
  visitor.visit(
      CudaMetadataFields::kChannelType,
      static_cast<int64_t>(activity.channelType));
#endif
}

inline void addBandwidthTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor,
    uint64_t bytes,
    int64_t duration) {
  if (duration == 0) {
    return;
  }
  visitor.visit(
      CudaMetadataFields::kMemoryBandwidthGbps,
      static_cast<double>(bytes) / static_cast<double>(duration));
}

// Convert limitingFactors bitmask to human-readable string
// Based on cudaOccLimitingFactor enum from cuda_occupancy.h
// This can be found in the CUDA toolkit typically
// /usr/local/cuda/targets/x86_64-linux/include/cuda_occupancy.h
inline std::string limitingFactorsToString(unsigned int factors) {
  if (factors == 0) {
    return "none";
  }
  constexpr std::pair<unsigned int, const char*> kFactors[] = {
      {OCC_LIMIT_WARPS, "WARPS"},
      {OCC_LIMIT_REGISTERS, "REGS"},
      {OCC_LIMIT_SHARED_MEMORY, "SMEM"},
      {OCC_LIMIT_BLOCKS, "BLOCKS"},
      {OCC_LIMIT_BARRIERS, "BARRIERS"},
  };
  std::string result;
  for (const auto& [mask, name] : kFactors) {
    if (factors & mask) {
      if (!result.empty()) {
        result += "|";
      }
      result += name;
    }
  }
  return result;
}

template <>
inline void GpuActivity<CUpti_ActivityKernelType>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivityKernelType& kernel = raw();
  const float blocksPerSmVal = blocksPerSm(kernel);
  const float warpsPerSmVal = warpsPerSm(kernel);
  const OccupancyMetrics occMetrics = computeOccupancyMetrics(kernel);

  visitor.visit(
      CudaMetadataFields::kQueued, static_cast<uint64_t>(kernel.queued));
  visitor.visit(
      CudaMetadataFields::kDevice, static_cast<int64_t>(kernel.deviceId));
  visitor.visit(
      CudaMetadataFields::kContext, static_cast<uint64_t>(kernel.contextId));
  visitor.visit(
      CudaMetadataFields::kStream, static_cast<int64_t>(kernel.streamId));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(kernel.correlationId));
  visitor.visit(
      CudaMetadataFields::kRegistersPerThread,
      static_cast<uint64_t>(kernel.registersPerThread));
  visitor.visit(
      CudaMetadataFields::kSharedMemory,
      static_cast<int64_t>(
          kernel.staticSharedMemory + kernel.dynamicSharedMemory));
  visitor.visit(
      CudaMetadataFields::kBlocksPerSm, static_cast<double>(blocksPerSmVal));
  visitor.visit(
      CudaMetadataFields::kWarpsPerSm, static_cast<double>(warpsPerSmVal));
  visitor.visit(
      CudaMetadataFields::kGrid,
      std::vector<int64_t>{
          static_cast<int64_t>(kernel.gridX),
          static_cast<int64_t>(kernel.gridY),
          static_cast<int64_t>(kernel.gridZ)});
  visitor.visit(
      CudaMetadataFields::kBlock,
      std::vector<int64_t>{
          static_cast<int64_t>(kernel.blockX),
          static_cast<int64_t>(kernel.blockY),
          static_cast<int64_t>(kernel.blockZ)});
  visitor.visit(
      CudaMetadataFields::kEstAchievedOccupancyPercent,
      static_cast<int64_t>(std::lround(occMetrics.occupancy * 100.0)));
  visitor.visit(CudaMetadataFields::kOccupancy, [&](auto& d) {
    d.visit(
        CudaMetadataFields::kActiveBlocksPerMultiprocessor,
        static_cast<int64_t>(occMetrics.result.activeBlocksPerMultiprocessor));
    d.visit(
        CudaMetadataFields::kLimitingFactors,
        limitingFactorsToString(occMetrics.result.limitingFactors));
    d.visit(
        CudaMetadataFields::kBlockLimitRegs,
        static_cast<int64_t>(occMetrics.result.blockLimitRegs));
    d.visit(
        CudaMetadataFields::kBlockLimitSharedMem,
        static_cast<int64_t>(occMetrics.result.blockLimitSharedMem));
    d.visit(
        CudaMetadataFields::kBlockLimitWarps,
        static_cast<int64_t>(occMetrics.result.blockLimitWarps));
    d.visit(
        CudaMetadataFields::kBlockLimitBlocks,
        static_cast<int64_t>(occMetrics.result.blockLimitBlocks));
    d.visit(
        CudaMetadataFields::kBlockLimitBarriers,
        static_cast<int64_t>(occMetrics.result.blockLimitBarriers));
    d.visit(
        CudaMetadataFields::kAllocatedRegistersPerBlock,
        static_cast<int64_t>(occMetrics.result.allocatedRegistersPerBlock));
    d.visit(
        CudaMetadataFields::kAllocatedSharedMemPerBlock,
        static_cast<uint64_t>(occMetrics.result.allocatedSharedMemPerBlock));
  });
  addGraphNodeTypedMetadata(visitor, kernel);
  addPriorityTypedMetadata(visitor, kernel);
  addChannelTypedMetadata(visitor, kernel);
}

template <>
inline const std::string GpuActivity<CUpti_ActivityKernelType>::metadataJson()
    const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline std::string memcpyName(uint8_t kind, uint8_t src, uint8_t dst) {
  return fmt::format(
      "Memcpy {} ({} -> {})",
      memcpyKindString((CUpti_ActivityMemcpyKind)kind),
      memoryKindString((CUpti_ActivityMemoryKind)src),
      memoryKindString((CUpti_ActivityMemoryKind)dst));
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemcpyType>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyType>::name() const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

template <>
inline void GpuActivity<CUpti_ActivityMemcpyType>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivityMemcpyType& memcpy = raw();
  visitor.visit(
      CudaMetadataFields::kDevice, static_cast<int64_t>(memcpy.deviceId));
  visitor.visit(
      CudaMetadataFields::kContext, static_cast<uint64_t>(memcpy.contextId));
  visitor.visit(
      CudaMetadataFields::kStream, static_cast<int64_t>(memcpy.streamId));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(memcpy.correlationId));
  visitor.visit(
      CudaMetadataFields::kBytes, static_cast<uint64_t>(memcpy.bytes));
  addBandwidthTypedMetadata(visitor, memcpy.bytes, duration());
  addGraphNodeTypedMetadata(visitor, memcpy);
  addChannelTypedMetadata(visitor, memcpy);
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyType>::metadataJson()
    const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemcpyPtoPType>::type() const {
  return ActivityType::GPU_MEMCPY;
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemcpyPtoPType>::name()
    const {
  return memcpyName(raw().copyKind, raw().srcKind, raw().dstKind);
}

template <>
inline void GpuActivity<CUpti_ActivityMemcpyPtoPType>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivityMemcpyPtoPType& memcpy = raw();
  visitor.visit(
      CudaMetadataFields::kFromDevice,
      static_cast<uint64_t>(memcpy.srcDeviceId));
  visitor.visit(
      CudaMetadataFields::kInDevice, static_cast<uint64_t>(memcpy.deviceId));
  visitor.visit(
      CudaMetadataFields::kToDevice, static_cast<uint64_t>(memcpy.dstDeviceId));
  visitor.visit(
      CudaMetadataFields::kFromContext,
      static_cast<uint64_t>(memcpy.srcContextId));
  visitor.visit(
      CudaMetadataFields::kInContext, static_cast<uint64_t>(memcpy.contextId));
  visitor.visit(
      CudaMetadataFields::kToContext,
      static_cast<uint64_t>(memcpy.dstContextId));
  visitor.visit(
      CudaMetadataFields::kStream, static_cast<int64_t>(memcpy.streamId));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(memcpy.correlationId));
  visitor.visit(
      CudaMetadataFields::kBytes, static_cast<uint64_t>(memcpy.bytes));
  addBandwidthTypedMetadata(visitor, memcpy.bytes, duration());
  addGraphNodeTypedMetadata(visitor, memcpy);
  addChannelTypedMetadata(visitor, memcpy);
}

template <>
inline const std::string GpuActivity<
    CUpti_ActivityMemcpyPtoPType>::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemsetType>::name() const {
  const char* memory_kind =
      memoryKindString((CUpti_ActivityMemoryKind)raw().memoryKind);
  return fmt::format("Memset ({})", memory_kind);
}

template <>
inline ActivityType GpuActivity<CUpti_ActivityMemsetType>::type() const {
  return ActivityType::GPU_MEMSET;
}

template <>
inline void GpuActivity<CUpti_ActivityMemsetType>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const CUpti_ActivityMemsetType& memset = raw();
  visitor.visit(
      CudaMetadataFields::kDevice, static_cast<int64_t>(memset.deviceId));
  visitor.visit(
      CudaMetadataFields::kContext, static_cast<uint64_t>(memset.contextId));
  visitor.visit(
      CudaMetadataFields::kStream, static_cast<int64_t>(memset.streamId));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(memset.correlationId));
  visitor.visit(
      CudaMetadataFields::kBytes, static_cast<uint64_t>(memset.bytes));
  addBandwidthTypedMetadata(visitor, memset.bytes, duration());
  addGraphNodeTypedMetadata(visitor, memset);
  addChannelTypedMetadata(visitor, memset);
}

template <>
inline const std::string GpuActivity<CUpti_ActivityMemsetType>::metadataJson()
    const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void DriverActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void OverheadActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline bool OverheadActivity::flowStart() const {
  return false;
}

inline const std::string OverheadActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void OverheadActivity::visitTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor) const {}

inline bool RuntimeActivity::flowStart() const {
  return CuptiCbidRegistry::instance().requiresFlowCorrelation(
      CallbackDomain::RUNTIME, activity_.cbid);
}

inline const std::string RuntimeActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void RuntimeActivity::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  visitor.visit(
      CudaMetadataFields::kCbid, static_cast<uint64_t>(activity_.cbid));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(activity_.correlationId));
}

inline bool isTrackedDriverCbid(const CUpti_ActivityAPI& activity_) {
  return CuptiCbidRegistry::instance().isRegistered(
      CallbackDomain::DRIVER, activity_.cbid);
}

inline bool DriverActivity::flowStart() const {
  return CuptiCbidRegistry::instance().requiresFlowCorrelation(
      CallbackDomain::DRIVER, activity_.cbid);
}

inline const std::string DriverActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void DriverActivity::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  visitor.visit(
      CudaMetadataFields::kCbid, static_cast<uint64_t>(activity_.cbid));
  visitor.visit(
      CudaMetadataFields::kCorrelation,
      static_cast<uint64_t>(activity_.correlationId));
}

inline const std::string DriverActivity::name() const {
  return driverCbidName(activity_.cbid);
}

template <class T>
inline const std::string GpuActivity<T>::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <class T>
inline void GpuActivity<T>::visitTypedMetadata(
    [[maybe_unused]] ITypedMetadataVisitor& visitor) const {}

} // namespace KINETO_NAMESPACE
