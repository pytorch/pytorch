/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XpuptiMetricQuery.h"

#include "ThrowUtil.h"
#include "XpuptiProfilerMacros.h"
#include "XpuptiScopeProfilerApi.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <pti/pti_metrics.h>

namespace libkineto {

namespace {

static_assert(kXpuDeviceUuidSize == PTI_MAX_DEVICE_UUID_SIZE);
constexpr std::string_view kMetricsHint =
    "Querying XPU metrics may require ZET_ENABLE_METRICS=1 in the environment";

} // namespace

std::vector<XpuMetricGroupNames> xpuptiAvailableMetrics(
    const XpuDeviceUuid& deviceUuid) {
  const auto devices = getMetricsDevices(kMetricsHint);

  // PTI device order is not guaranteed to match torch.xpu device indices.
  const pti_device_properties_t* device = nullptr;
  for (const auto& candidate : devices) {
    if (std::memcmp(candidate._uuid, deviceUuid.data(), deviceUuid.size()) ==
        0) {
      device = &candidate;
      break;
    }
  }
  if (device == nullptr) {
    KINETO_THROW(
        std::runtime_error,
        fmt::format(
            "The requested XPU device does not support metrics collection "
            "(PTI reports {} metrics-capable device(s))",
            devices.size()));
  }

  uint32_t groupCount = 0;
  XPUPTI_CALL(ptiMetricsGetMetricGroups(device->_handle, nullptr, &groupCount));
  auto groups = std::make_unique<pti_metrics_group_properties_t[]>(groupCount);
  XPUPTI_CALL(
      ptiMetricsGetMetricGroups(device->_handle, groups.get(), &groupCount));

  // Only event-based groups can be collected by the XPUPTI scope profiler, and
  // metrics collected together must all belong to one group.
  std::vector<XpuMetricGroupNames> result;
  for (uint32_t g = 0; g < groupCount; ++g) {
    const auto& group = groups[g];
    if (group._type != PTI_METRIC_GROUP_TYPE_EVENT_BASED ||
        group._metric_count == 0) {
      continue;
    }
    auto metrics =
        std::make_unique<pti_metric_properties_t[]>(group._metric_count);
    XPUPTI_CALL(ptiMetricsGetMetricsProperties(group._handle, metrics.get()));
    auto& [groupName, metricNames] = result.emplace_back();
    groupName = group._name;
    metricNames.reserve(group._metric_count);
    for (uint32_t m = 0; m < group._metric_count; ++m) {
      metricNames.emplace_back(metrics[m]._name);
    }
  }
  return result;
}

} // namespace libkineto
