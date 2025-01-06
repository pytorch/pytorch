#include <ATen/xpu/XPUContext.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

#include <deque>
#include <vector>

namespace at::xpu {
namespace {

/*
 * Currently, there is one device properties pool containing the information and
 * capability about each compute-device.
 *
 * Device properties are lazily initialized when the first time properties are
 * requested for a device.
 */
DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_prop_flags;
std::vector<DeviceProp> device_properties;

std::deque<c10::once_flag> device_global_idx_flags;
std::vector<int32_t> device_global_idxs;

void initXPUContextVectors() {
  num_gpus = c10::xpu::device_count();
  device_prop_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
  device_global_idx_flags.resize(num_gpus);
  device_global_idxs.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device) {
  c10::xpu::get_device_properties(&device_properties[device], device);
}

void initDeviceGlobalIdx(DeviceIndex device) {
  sycl::device& raw_device = c10::xpu::get_raw_device(device);
  // Get all SYCL devices associated with the SYCL platform.
  auto devices = sycl::device::get_devices();
  auto match_device = [raw_device](const auto& dev) -> bool {
    return raw_device == dev;
  };
  auto it = std::find_if(devices.begin(), devices.end(), match_device);
  TORCH_CHECK(
      it != devices.end(), "Can't find the global index of XPU device.");
  device_global_idxs[device] =
      static_cast<int32_t>(std::distance(devices.begin(), it));
}

} // anonymous namespace

DeviceProp* getCurrentDeviceProperties() {
  auto device = c10::xpu::current_device();
  return getDeviceProperties(device);
}

DeviceProp* getDeviceProperties(DeviceIndex device) {
  c10::call_once(init_flag, initXPUContextVectors);
  if (device == -1)
    device = c10::xpu::current_device();
  check_device_index(device);
  c10::call_once(device_prop_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

// Return the global index enumerated by sycl::device::get_devices based on the
// index of a XPU device in the framework.
int32_t getGlobalIdxFromDevice(DeviceIndex device) {
  c10::call_once(init_flag, initXPUContextVectors);
  check_device_index(device);
  c10::call_once(device_global_idx_flags[device], initDeviceGlobalIdx, device);
  return device_global_idxs[device];
}

} // namespace at::xpu
