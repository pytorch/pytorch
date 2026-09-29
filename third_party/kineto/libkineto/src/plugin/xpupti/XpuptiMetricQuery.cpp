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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>
#include <pti/pti_metrics.h>

namespace libkineto {

namespace {

constexpr std::string_view kMetricsHint =
    "Querying XPU metrics may require ZET_ENABLE_METRICS=1 in the environment";

const char* metricTypeToString(pti_metric_type type) {
  switch (type) {
    case PTI_METRIC_TYPE_DURATION:
      return "duration";
    case PTI_METRIC_TYPE_EVENT:
      return "event";
    case PTI_METRIC_TYPE_EVENT_WITH_RANGE:
      return "event_with_range";
    case PTI_METRIC_TYPE_THROUGHPUT:
      return "throughput";
    case PTI_METRIC_TYPE_TIMESTAMP:
      return "timestamp";
    case PTI_METRIC_TYPE_FLAG:
      return "flag";
    case PTI_METRIC_TYPE_RATIO:
      return "ratio";
    case PTI_METRIC_TYPE_RAW:
      return "raw";
    case PTI_METRIC_TYPE_IP:
      return "ip";
    default:
      return "unknown";
  }
}

const char* metricValueTypeToString(pti_metric_value_type type) {
  switch (type) {
    case PTI_METRIC_VALUE_TYPE_UINT32:
      return "uint32";
    case PTI_METRIC_VALUE_TYPE_UINT64:
      return "uint64";
    case PTI_METRIC_VALUE_TYPE_FLOAT32:
      return "float32";
    case PTI_METRIC_VALUE_TYPE_FLOAT64:
      return "float64";
    case PTI_METRIC_VALUE_TYPE_BOOL8:
      return "bool8";
    case PTI_METRIC_VALUE_TYPE_STRING:
      return "string";
    case PTI_METRIC_VALUE_TYPE_UINT8:
      return "uint8";
    case PTI_METRIC_VALUE_TYPE_UINT16:
      return "uint16";
    default:
      return "unknown";
  }
}

} // namespace

std::vector<XpuMetricGroupInfo> xpuptiAvailableMetrics(
    const std::array<uint8_t, 16>& deviceUuid) {
  uint32_t deviceCount = 0;
  XPUPTI_CALL(ptiMetricsGetDevices(nullptr, &deviceCount), kMetricsHint);
  auto devices = std::make_unique<pti_device_properties_t[]>(deviceCount);
  XPUPTI_CALL(ptiMetricsGetDevices(devices.get(), &deviceCount), kMetricsHint);

  // PTI device order is not guaranteed to match torch.xpu device indices.
  static_assert(PTI_MAX_DEVICE_UUID_SIZE == 16);
  const pti_device_properties_t* device = nullptr;
  for (uint32_t i = 0; i < deviceCount; ++i) {
    if (std::memcmp(devices[i]._uuid, deviceUuid.data(), deviceUuid.size()) ==
        0) {
      device = &devices[i];
      break;
    }
  }
  if (device == nullptr) {
    KINETO_THROW(
        std::runtime_error,
        fmt::format(
            "The requested XPU device does not support metrics collection "
            "(PTI reports {} metrics-capable device(s))",
            deviceCount));
  }

  uint32_t groupCount = 0;
  XPUPTI_CALL(ptiMetricsGetMetricGroups(device->_handle, nullptr, &groupCount));
  auto groups = std::make_unique<pti_metrics_group_properties_t[]>(groupCount);
  XPUPTI_CALL(
      ptiMetricsGetMetricGroups(device->_handle, groups.get(), &groupCount));

  std::vector<XpuMetricGroupInfo> result;
  result.reserve(groupCount);
  for (uint32_t g = 0; g < groupCount; ++g) {
    const auto& group = groups[g];
    XpuMetricGroupInfo& groupInfo = result.emplace_back();
    groupInfo.name = group._name;
    groupInfo.description = group._description;
    // Only event-based groups can be collected by the XPUPTI scope profiler.
    groupInfo.scopeCompatible =
        group._type == PTI_METRIC_GROUP_TYPE_EVENT_BASED;
    if (group._metric_count == 0) {
      continue;
    }

    auto metrics =
        std::make_unique<pti_metric_properties_t[]>(group._metric_count);
    XPUPTI_CALL(ptiMetricsGetMetricsProperties(group._handle, metrics.get()));
    groupInfo.metrics.reserve(group._metric_count);
    for (uint32_t m = 0; m < group._metric_count; ++m) {
      const auto& metric = metrics[m];
      groupInfo.metrics.push_back(
          {metric._name,
           metric._description,
           metric._units,
           metricTypeToString(metric._metric_type),
           metricValueTypeToString(metric._value_type)});
    }
  }
  return result;
}

} // namespace libkineto
