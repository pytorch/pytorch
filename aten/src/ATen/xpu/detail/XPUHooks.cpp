#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUDevice.h>
#include <ATen/xpu/detail/XPUHooks.h>
#include <c10/util/CallOnce.h>

namespace at::xpu::detail {

void XPUHooks::initXPU() const {
  // TODO: the initialization of device allocator should be placed here.
}

bool XPUHooks::hasXPU() const {
  return true;
}

std::string XPUHooks::showConfig() const {
  return "XPU backend";
}

int XPUHooks::getGlobalIdxFromDevice(const at::Device& device) const {
  TORCH_CHECK(device.is_xpu(), "Only the XPU device type is expected.");
  return at::xpu::getGlobalIdxFromDevice(device.index());
}

Device XPUHooks::getDeviceFromPtr(void* data) const {
  return at::xpu::getDeviceFromPtr(data);
}

int XPUHooks::getNumGPUs() const {
  return at::xpu::device_count();
}

REGISTER_XPU_HOOKS(XPUHooks);

} // namespace at::xpu::detail
