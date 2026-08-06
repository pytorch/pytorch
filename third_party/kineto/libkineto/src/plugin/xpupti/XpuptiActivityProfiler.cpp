/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiActivityProfiler.h"
#include "XpuptiScopeProfilerApi.h"
#include "XpuptiScopeProfilerSession.h"

#include <fmt/ranges.h>
#include <sycl/sycl.hpp>

namespace KINETO_NAMESPACE {

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
      jsonProps.push_back(fmt::format(
          R"JSON(
    {{
      "id": {},
      "name": "{}",
      "totalGlobalMem": {},
      "maxComputeUnits": {},
      "maxWorkGroupSize": {},
      "maxClockFrequency": {},
      "maxMemAllocSize": {},
      "localMemSize": {},
      "vendor": "{}",
      "driverVersion": "{}"
    }})JSON",
          i,
          device.get_info<sycl::info::device::name>(),
          device.get_info<sycl::info::device::global_mem_size>(),
          device.get_info<sycl::info::device::max_compute_units>(),
          device.get_info<sycl::info::device::max_work_group_size>(),
          device.get_info<sycl::info::device::max_clock_frequency>(),
          device.get_info<sycl::info::device::max_mem_alloc_size>(),
          device.get_info<sycl::info::device::local_mem_size>(),
          device.get_info<sycl::info::device::vendor>(),
          device.get_info<sycl::info::device::driver_version>()));
    }
  }

  return fmt::format("{}", fmt::join(jsonProps, ","));
}

[[noreturn]] const std::set<ActivityType>& XPUActivityProfiler::
    availableActivities() const {
  throw std::runtime_error(
      "The availableActivities is legacy method and should not be called by kineto");
}

std::unique_ptr<libkineto::IActivityProfilerSession> XPUActivityProfiler::
    configure(
        const std::set<ActivityType>& activity_types,
        const libkineto::Config& config) {
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
