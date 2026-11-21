#ifdef HAS_XPUPTI
#include <torch/csrc/xpu/MetricsQuery.h>

#include <ATen/xpu/XPUContext.h>
#include <c10/util/Exception.h>

#include <pti/pti.h>
#include <pti/pti_metrics.h>

#include <cstring>
#include <memory>
#include <tuple>

namespace torch::xpu {

namespace {

constexpr const char* kMetricsHint =
    "Querying XPU metrics may require ZET_ENABLE_METRICS=1 in the environment";

// hint tells the user how to fix the most likely cause of the failure.
inline void checkPtiResult(
    pti_result result,
    const char* ptiFunction,
    const char* hint) {
  TORCH_CHECK(
      result == PTI_SUCCESS,
      ptiFunction,
      " failed with ",
      ptiResultTypeToString(result),
      ". ",
      hint);
}

// Devices PTI can collect metrics on.
std::vector<pti_device_properties_t> getMetricsDevices() {
  uint32_t deviceCount = 0;
  checkPtiResult(
      ptiMetricsGetDevices(nullptr, &deviceCount),
      "ptiMetricsGetDevices",
      kMetricsHint);
  if (deviceCount == 0) {
    return {};
  }
  std::vector<pti_device_properties_t> devices(deviceCount);
  checkPtiResult(
      ptiMetricsGetDevices(devices.data(), &deviceCount),
      "ptiMetricsGetDevices",
      kMetricsHint);
  devices.resize(deviceCount);
  return devices;
}

} // namespace

std::vector<MetricGroupNames> availableScopeProfilerMetrics(
    c10::DeviceIndex device) {
  // The memcmp below assumes both UUID representations have the same size.
  static_assert(
      std::tuple_size_v<decltype(c10::xpu::DeviceProp::uuid)> ==
      PTI_MAX_DEVICE_UUID_SIZE);
  const auto& uuid = at::xpu::getDeviceProperties(device)->uuid;

  // PTI device order is not guaranteed to match torch.xpu device indices.
  const auto devices = getMetricsDevices();
  const pti_device_properties_t* ptiDevice = nullptr;
  for (const auto& candidate : devices) {
    if (std::memcmp(candidate._uuid, uuid.data(), uuid.size()) == 0) {
      ptiDevice = &candidate;
      break;
    }
  }
  TORCH_CHECK(
      ptiDevice != nullptr,
      "The requested XPU device does not support metrics collection "
      "(PTI reports ",
      devices.size(),
      " metrics-capable device(s)). ",
      kMetricsHint);

  uint32_t groupCount = 0;
  checkPtiResult(
      ptiMetricsGetMetricGroups(ptiDevice->_handle, nullptr, &groupCount),
      "ptiMetricsGetMetricGroups",
      kMetricsHint);
  auto groups = std::make_unique<pti_metrics_group_properties_t[]>(groupCount);
  checkPtiResult(
      ptiMetricsGetMetricGroups(ptiDevice->_handle, groups.get(), &groupCount),
      "ptiMetricsGetMetricGroups",
      kMetricsHint);

  // Only event-based groups can be collected by the XPUPTI scope profiler, and
  // metrics collected together must all belong to one group.
  std::vector<MetricGroupNames> result;
  for (uint32_t g = 0; g < groupCount; ++g) {
    const auto& group = groups[g];
    if (group._type != PTI_METRIC_GROUP_TYPE_EVENT_BASED ||
        group._metric_count == 0) {
      continue;
    }
    auto metrics =
        std::make_unique<pti_metric_properties_t[]>(group._metric_count);
    checkPtiResult(
        ptiMetricsGetMetricsProperties(group._handle, metrics.get()),
        "ptiMetricsGetMetricsProperties",
        kMetricsHint);
    auto& [groupName, metricNames] = result.emplace_back();
    groupName = group._name;
    metricNames.reserve(group._metric_count);
    for (uint32_t m = 0; m < group._metric_count; ++m) {
      metricNames.emplace_back(metrics[m]._name);
    }
  }
  return result;
}

} // namespace torch::xpu
#endif // HAS_XPUPTI
