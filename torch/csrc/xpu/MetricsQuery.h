#pragma once

#include <c10/core/Device.h>
#include <torch/csrc/Export.h>

#include <string>
#include <utility>
#include <vector>

namespace torch::xpu {

// A metric group's name and the names of its metrics.
using MetricGroupNames = std::pair<std::string, std::vector<std::string>>;

// Event-based PTI metric groups the XPUPTI scope profiler can collect on
// `device`. Only available when PyTorch is built with Kineto's XPUPTI backend.
TORCH_PYTHON_API std::vector<MetricGroupNames> availableScopeProfilerMetrics(
    c10::DeviceIndex device);

} // namespace torch::xpu
