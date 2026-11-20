/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfiler.h"
#include "Logger.h"
#include "MetadataFieldCatalog.h"
#include "ThrowUtil.h"
#include "TypedMetadata.h"
#include "TypedMetadataJson.h"
#include "XpuptiActivityApi.h"
#include "XpuptiScopeProfilerSession.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <pti/pti.h>
#include <sycl/sycl.hpp>

namespace KINETO_NAMESPACE {

namespace DevicePropertyFields = libkineto::DevicePropertyMetadataFields;

namespace {

void visitXpuDeviceMetadata(
    size_t id,
    const sycl::device& device,
    libkineto::ITypedMetadataVisitor& visitor) {
  visitor.visit(DevicePropertyFields::kId, static_cast<uint64_t>(id));
  visitor.visit(
      DevicePropertyFields::kName, device.get_info<sycl::info::device::name>());
  visitor.visit(
      DevicePropertyFields::kTotalGlobalMem,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::global_mem_size>()));
  visitor.visit(
      DevicePropertyFields::kMaxComputeUnits,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::max_compute_units>()));
  visitor.visit(
      DevicePropertyFields::kMaxWorkGroupSize,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::max_work_group_size>()));
  visitor.visit(
      DevicePropertyFields::kMaxClockFrequency,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::max_clock_frequency>()));
  visitor.visit(
      DevicePropertyFields::kMaxMemAllocSize,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::max_mem_alloc_size>()));
  visitor.visit(
      DevicePropertyFields::kLocalMemSize,
      static_cast<uint64_t>(
          device.get_info<sycl::info::device::local_mem_size>()));
  visitor.visit(
      DevicePropertyFields::kVendor,
      device.get_info<sycl::info::device::vendor>());
  visitor.visit(
      DevicePropertyFields::kDriverVersion,
      device.get_info<sycl::info::device::driver_version>());

  // CUDA-named aliases, only where the meaning matches: an SM is an Xe-core
  // (maxComputeUnits counts EUs). warpSize and maxThreadsPerMultiprocessor
  // depend on the kernel's sub-group size and GRF mode, so they are omitted.
  const auto localMemSize = static_cast<uint64_t>(
      device.get_info<sycl::info::device::local_mem_size>());
  visitor.visit(
      DevicePropertyFields::kMaxThreadsPerBlock,
      static_cast<int64_t>(
          device.get_info<sycl::info::device::max_work_group_size>()));
  visitor.visit(DevicePropertyFields::kSharedMemPerBlock, localMemSize);
  visitor.visit(DevicePropertyFields::kSharedMemPerBlockOptin, localMemSize);

  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  visitor.visit(
      DevicePropertyFields::kSubGroupSizes,
      std::vector<int64_t>(sizes.begin(), sizes.end()));

  namespace intel_info = sycl::ext::intel::info::device;
  if (device.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
    const auto eusPerXeCore =
        device.get_info<intel_info::gpu_eu_count_per_subslice>();
    visitor.visit(
        DevicePropertyFields::kEusPerXeCore,
        static_cast<uint64_t>(eusPerXeCore));
    // Same formula as torch.xpu's gpu_subslice_count.
    if (device.has(sycl::aspect::ext_intel_gpu_eu_count) && eusPerXeCore > 0) {
      const auto numXeCores =
          device.get_info<intel_info::gpu_eu_count>() / eusPerXeCore;
      visitor.visit(
          DevicePropertyFields::kNumXeCores, static_cast<uint64_t>(numXeCores));
      visitor.visit(
          DevicePropertyFields::kNumSms, static_cast<int64_t>(numXeCores));
    }
  }
  if (device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    visitor.visit(
        DevicePropertyFields::kHwThreadsPerEu,
        static_cast<uint64_t>(
            device.get_info<intel_info::gpu_hw_threads_per_eu>()));
  }
  if (device.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
    visitor.visit(
        DevicePropertyFields::kSimdWidth,
        static_cast<uint64_t>(
            device.get_info<intel_info::gpu_eu_simd_width>()));
  }
}

} // namespace

std::string getXpuDeviceProperties() {
  std::vector<std::string> jsonProps;
  // Enumerated GPU devices from the specific platform
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    const auto& device_list = platform.get_devices();
    for (size_t i = 0; i < device_list.size(); i++) {
      const auto& device = device_list[i];
      libkineto::internal::JsonTypedMetadataVisitor visitor;
      visitXpuDeviceMetadata(i, device, visitor);
      jsonProps.push_back("{" + std::move(visitor).json() + "}");
    }
  }

  return fmt::format("{}", fmt::join(jsonProps, ","));
}

[[noreturn]] const std::set<ActivityType>& XPUActivityProfiler::
    availableActivities() const {
  KINETO_THROW(
      std::runtime_error,
      "The availableActivities is legacy method and should not be called by kineto");
}

std::unique_ptr<libkineto::IActivityProfilerSession> XPUActivityProfiler::
    configure(
        const std::set<ActivityType>& activity_types,
        const libkineto::Config& config) {
  if (!config.deviceTracingEnabled()) {
    return nullptr;
  }
  // Without PTI, the trace goes on without XPU activities, as a CUDA trace
  // does without CUPTI. Raising here would end the process for an on-demand
  // trace, which configures on the profiler thread.
  if (const pti_result result = ptiViewGPULocalAvailable();
      result != PTI_SUCCESS) {
    LOG(ERROR) << "Failed to enable Kineto Profiler on XPU: "
               << ptiResultTypeToString(result) << " ("
               << static_cast<int>(result) << ")";
    return nullptr;
  }
  return std::make_unique<XpuptiScopeProfilerSession>(
      XpuptiActivityApi::singleton(), name(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession> XPUActivityProfiler::
    configure(
        [[maybe_unused]] int64_t ts_ms,
        [[maybe_unused]] int64_t duration_ms,
        const std::set<ActivityType>& activity_types,
        const libkineto::Config& config) {
  return configure(activity_types, config);
}

} // namespace KINETO_NAMESPACE
