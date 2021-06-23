#include "lazy_tensor_core/csrc/device.h"

#include <c10/util/Optional.h>

#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/str_cat.h"
#include "lazy_tensors/str_split.h"

namespace torch_lazy_tensors {
namespace {

thread_local c10::optional<Device> g_current_device;

std::string DeviceTypeToString(DeviceType hw_type) {
  switch (hw_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::GPU:
      return "GPU";
    case DeviceType::TPU:
      return "TPU";
  }
  LTC_ERROR() << "Invalid device type";
}

void ParseDevice(const std::string& device_spec, Device* device) {
  if (device_spec.empty()) {
    std::string default_device_spec =
        lazy_tensors::ComputationClient::Get()->GetDefaultDevice();
    LTC_CHECK(!default_device_spec.empty());
    return ParseDevice(default_device_spec, device);
  }
  if (device_spec[0] == ':') {
    std::string default_device_spec =
        lazy_tensors::ComputationClient::Get()->GetDefaultDevice();
    auto pos = default_device_spec.find(':');
    LTC_CHECK_NE(pos, std::string::npos) << default_device_spec;
    return ParseDevice(default_device_spec.substr(0, pos) + device_spec,
                       device);
  }
  std::vector<std::string> device_spec_parts =
      lazy_tensors::StrSplit(device_spec, ':');
  LTC_CHECK_EQ(device_spec_parts.size(), 2)
      << "Invalid device specification: " << device_spec;

  device->ordinal = std::stoi(device_spec_parts[1]);
  if (device_spec_parts[0] == "TPU") {
    device->hw_type = DeviceType::TPU;
  } else if (device_spec_parts[0] == "CPU") {
    device->hw_type = DeviceType::CPU;
  } else if (device_spec_parts[0] == "GPU") {
    device->hw_type = DeviceType::GPU;
  } else {
    LTC_ERROR() << "Invalid device specification: " << device_spec;
  }
}

}  // namespace

Device::Device(const std::string& device_spec) {
  ParseDevice(device_spec, this);
}

std::string Device::ToString() const {
  return lazy_tensors::StrCat(DeviceTypeToString(hw_type), ":", ordinal);
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
  return default_device;
}

Device GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

Device SetCurrentDevice(const Device& device) {
  Device current = GetCurrentDevice();
  g_current_device = device;
  LTC_VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace torch_lazy_tensors
