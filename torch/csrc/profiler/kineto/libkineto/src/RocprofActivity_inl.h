/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "RocprofActivity.h"

#include <fmt/format.h>
#include <stddef.h>
#include <cstdint>
#include <string>
#include <vector>

#include "Demangle.h"
#include "TypedMetadataJson.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

namespace {
thread_local std::unordered_map<int, std::vector<int64_t>> correlationToGrid;
thread_local std::unordered_map<int, std::vector<int64_t>> correlationToBlock;
thread_local std::unordered_map<int, size_t> correlationToSize;

inline std::vector<int64_t> rocprofKernelGrid(
    const rocprofKernelRow& activity) {
  return {
      static_cast<int64_t>(activity.gridX),
      static_cast<int64_t>(activity.gridY),
      static_cast<int64_t>(activity.gridZ)};
}

inline std::vector<int64_t> rocprofKernelBlock(
    const rocprofKernelRow& activity) {
  return {
      static_cast<int64_t>(activity.workgroupX),
      static_cast<int64_t>(activity.workgroupY),
      static_cast<int64_t>(activity.workgroupZ)};
}

} // namespace

inline const char* getGpuActivityKindString(uint32_t domain, uint32_t op) {
  if (domain == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
    return "Dispatch Kernel";
  else if (domain == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
    switch (op) {
      case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:
        return "HtoH";
      case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:
        return "HtoD";
      case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
        return "DtoH";
      case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
        return "DtoD";
    }
  }
  return "<unknown>";
}

inline void getMemcpySrcDstString(
    uint32_t kind,
    std::string& src,
    std::string& dst) {
  switch (kind) {
    case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:
      src = "Host";
      dst = "Host";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
      src = "Device";
      dst = "Host";
      break;
    case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:
      src = "Host";
      dst = "Device";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
      src = "Device";
      dst = "Device";
      break;
    default:
      src = "?";
      dst = "?";
      break;
  }
}

// GPU Activities

inline const std::string GpuActivity::name() const {
  if (type_ == ActivityType::CONCURRENT_KERNEL) {
    auto op = raw().op;
    auto domain = raw().domain;
    std::string opString = RocprofLogger::opString(
        static_cast<rocprofiler_buffer_tracing_kind_t>(domain), op);
    const char* name = opString.c_str();
    return demangle(
        raw().kernelName.length() > 0 ? raw().kernelName : std::string(name));
  } else if (type_ == ActivityType::GPU_MEMSET) {
    return fmt::format(
        "Memset ({})", getGpuActivityKindString(raw().domain, raw().op));
  } else if (type_ == ActivityType::GPU_MEMCPY) {
    std::string src = "";
    std::string dst = "";
    getMemcpySrcDstString(raw().op, src, dst);
    return fmt::format(
        "Memcpy {} ({} -> {})",
        getGpuActivityKindString(raw().domain, raw().op),
        src,
        dst);
  } else {
    return "";
  }
  return "";
}

inline void GpuActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

static inline void addBandwidthTypedMetadata(
    ITypedMetadataVisitor& visitor,
    size_t bytes,
    uint64_t duration) {
  if (duration == 0) {
    return;
  }
  visitor.visit(
      RocmMetadataFields::kMemoryBandwidthGbps,
      static_cast<double>(bytes) / static_cast<double>(duration));
}

inline const std::string GpuActivity::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

inline void GpuActivity::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  const auto& gpuActivity = raw();
  visitor.visit(
      RocmMetadataFields::kDevice, static_cast<int64_t>(gpuActivity.device));
  visitor.visit(RocmMetadataFields::kStream, resourceId());
  visitor.visit(
      RocmMetadataFields::kHsaQueue, static_cast<uint64_t>(gpuActivity.queue));
  visitor.visit(
      RocmMetadataFields::kCorrelation, static_cast<uint64_t>(gpuActivity.id));
  visitor.visit(
      RocmMetadataFields::kKind,
      std::string{
          getGpuActivityKindString(gpuActivity.domain, gpuActivity.op)});

  // if memcpy or memset, add size
  auto sizeIt = correlationToSize.find(gpuActivity.id);
  if (sizeIt != correlationToSize.end()) {
    visitor.visit(
        RocmMetadataFields::kBytes, static_cast<uint64_t>(sizeIt->second));
    addBandwidthTypedMetadata(
        visitor, sizeIt->second, gpuActivity.end - gpuActivity.begin);
    return;
  }

  // if compute kernel, add grid and block
  auto gridIt = correlationToGrid.find(gpuActivity.id);
  auto blockIt = correlationToBlock.find(gpuActivity.id);
  if (gridIt != correlationToGrid.end() &&
      blockIt != correlationToBlock.end()) {
    visitor.visit(RocmMetadataFields::kGrid, gridIt->second);
    visitor.visit(RocmMetadataFields::kBlock, blockIt->second);
  }
}

// Runtime Activities

template <class T>
inline bool RuntimeActivity<T>::flowStart() const {
  bool should_correlate =
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipFree ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream ||
      raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch;
  return should_correlate;
}

template <class T>
inline void RuntimeActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

template <>
inline void RuntimeActivity<rocprofKernelRow>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  if ((raw().functionAddr != nullptr)) {
    visitor.visit(
        RocmMetadataFields::kKernel,
        demangle(hipKernelNameRefByPtr(raw().functionAddr, raw().stream)));
  } else if ((raw().function != nullptr)) {
    visitor.visit(
        RocmMetadataFields::kKernel,
        demangle(hipKernelNameRef(raw().function)));
  }

  // cache grid and block so we can pass it into async activity (GPU track)
  correlationToGrid[raw().id] = rocprofKernelGrid(raw());
  correlationToBlock[raw().id] = rocprofKernelBlock(raw());

  visitor.visit(RocmMetadataFields::kCid, static_cast<uint64_t>(raw().cid));
  visitor.visit(
      RocmMetadataFields::kCorrelation, static_cast<uint64_t>(raw().id));
  visitor.visit(RocmMetadataFields::kGrid, correlationToGrid[raw().id]);
  visitor.visit(RocmMetadataFields::kBlock, correlationToBlock[raw().id]);
  visitor.visit(
      RocmMetadataFields::kSharedMemory,
      static_cast<uint64_t>(raw().groupSegmentSize));
}

template <>
inline const std::string RuntimeActivity<rocprofKernelRow>::metadataJson()
    const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <>
inline void RuntimeActivity<rocprofCopyRow>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  correlationToSize[raw().id] = raw().size;
  visitor.visit(RocmMetadataFields::kCid, static_cast<uint64_t>(raw().cid));
  visitor.visit(
      RocmMetadataFields::kCorrelation, static_cast<uint64_t>(raw().id));
  visitor.visit(RocmMetadataFields::kSrc, fmt::format("{}", raw().src));
  visitor.visit(RocmMetadataFields::kDst, fmt::format("{}", raw().dst));
  visitor.visit(RocmMetadataFields::kBytes, static_cast<uint64_t>(raw().size));
  visitor.visit(
      RocmMetadataFields::kKind,
      fmt::format("{}", fmt::underlying(raw().kind)));
}

template <>
inline const std::string RuntimeActivity<rocprofCopyRow>::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <>
inline void RuntimeActivity<rocprofMallocRow>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  correlationToSize[raw().id] = raw().size;
  if (raw().cid == ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc) {
    visitor.visit(
        RocmMetadataFields::kBytes, static_cast<uint64_t>(raw().size));
  }
  visitor.visit(RocmMetadataFields::kCid, static_cast<uint64_t>(raw().cid));
  visitor.visit(
      RocmMetadataFields::kCorrelation, static_cast<uint64_t>(raw().id));
  visitor.visit(RocmMetadataFields::kPtr, fmt::format("{}", raw().ptr));
}

template <>
inline const std::string RuntimeActivity<rocprofMallocRow>::metadataJson()
    const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <class T>
inline const std::string RuntimeActivity<T>::metadataJson() const {
  libkineto::internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

template <class T>
inline void RuntimeActivity<T>::visitTypedMetadata(
    ITypedMetadataVisitor& visitor) const {
  visitor.visit(RocmMetadataFields::kCid, static_cast<uint64_t>(raw().cid));
  visitor.visit(
      RocmMetadataFields::kCorrelation, static_cast<uint64_t>(raw().id));
}

} // namespace KINETO_NAMESPACE
