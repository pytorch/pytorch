#include "Hooks.h"
#include "Generator.h"
#include "../vulkan/Context.h"

namespace torch_vulkan {

bool VulkanHooksInterface::hasPrimaryContext(c10::DeviceIndex /*device_index*/) const {
    return vulkan::Context::instance().is_available();
}

bool VulkanHooksInterface::isAvailable() const {
    return vulkan::Context::instance().is_available();
}

at::Generator VulkanHooksInterface::getNewGenerator(
    c10::DeviceIndex device_index) const {
    return MakeVulkanGenerator(device_index < 0 ? 0 : device_index);
}

const at::Generator& VulkanHooksInterface::getDefaultGenerator(
    c10::DeviceIndex device_index) const {
    static auto gen = MakeVulkanGenerator(device_index < 0 ? 0 : device_index);
    return gen;
}

} // namespace torch_vulkan
