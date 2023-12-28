#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUDevice.h>
#include <ATen/xpu/detail/XPUHooks.h>
#include <c10/util/CallOnce.h>

namespace at::xpu::detail {

void XPUHooks::initXPU() const {
  // TODO:
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "XPU backend";
}

int XPUHooks::getGlobalIdFromDevice(const at::Device& device) const {
  TORCH_CHECK(device.is_xpu(), "Only the XPU device type is expected.");
  return at::xpu::getGlobalIdFromDevice(device.index());
}

Device XPUHooks::getDeviceFromPtr(void* data) const {
  return at::xpu::getDeviceFromPtr(data);
}

int XPUHooks::getNumGPUs() const {
  return at::xpu::device_count();
}

void XPUHooks::deviceSynchronize(DeviceIndex device_index) const {
  c10::xpu::device_synchronize(device_index);
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace at::xpu::detail
