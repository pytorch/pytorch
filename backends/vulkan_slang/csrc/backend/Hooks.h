#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/core/Generator.h>

namespace torch_vulkan {

struct VulkanHooksInterface : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
    bool isBuilt() const override { return true; }
    bool isAvailable() const override;
    at::Generator getNewGenerator(
        c10::DeviceIndex device_index = -1) const override;
    const at::Generator& getDefaultGenerator(
        c10::DeviceIndex device_index) const override;
};

} // namespace torch_vulkan
