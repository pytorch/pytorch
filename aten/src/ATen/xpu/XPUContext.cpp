#include <ATen/xpu/XPUContext.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <deque>
#include <mutex>
#include <vector>

namespace at::xpu {
namespace {

/*
 * Currently, there is one device property pool containing the information and
 * capability about each compute-device.
 *
 * Device properties are lazily initialized when the first time properties are
 * requested for a device.
 */
DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_prop_flags;
std::vector<DeviceProp> device_properties;

std::deque<c10::once_flag> device_global_index_flags;
std::vector<int> device_global_indexs;

void initXPUContextVectors() {
  num_gpus = c10::xpu::device_count();
  device_prop_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
  device_global_index_flags.resize(num_gpus);
  device_global_indexs.resize(num_gpus);
}

void initDeviceProperty(int device) {
  DeviceProp device_prop;
  c10::xpu::get_device_properties(&device_prop, device);
  device_properties[device] = device_prop;
}

void initDeviceGlobalId(int device) {
  sycl::device& raw_device = c10::xpu::get_raw_device(device);
  // Get all SYCL devices associated with the SYCL platform.
  auto devices = sycl::device::get_devices();
  auto match_device = [raw_device](const auto& device) -> bool {
    return raw_device == device;
  };
  auto it = std::find_if(devices.begin(), devices.end(), match_device);
  TORCH_CHECK(it != devices.end(), "Cant't find the global id of XPU device.");
  device_global_indexs[device] =
      static_cast<int>(std::distance(devices.begin(), it));
}

} // anonymous namespace

DeviceProp* getCurrentDeviceProperties() {
  auto device = c10::xpu::current_device();
  return getDeviceProperties(device);
}

DeviceProp* getDeviceProperties(int device) {
  c10::call_once(init_flag, initXPUContextVectors);
  if (device == -1)
    device = c10::xpu::current_device();
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      device,
      ", total number of device is ",
      num_gpus,
      ".");
  c10::call_once(device_prop_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

int getGlobalIdxFromDevice(int device) {
  c10::call_once(init_flag, initXPUContextVectors);
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      device,
      ", total number of device is ",
      num_gpus,
      ".");
  c10::call_once(device_global_index_flags[device], initDeviceGlobalId, device);
  return device_global_indexs[device];
}

} // namespace at::xpu
