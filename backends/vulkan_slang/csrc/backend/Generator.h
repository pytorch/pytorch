#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>

namespace torch_vulkan {

struct VulkanGeneratorImpl : public c10::GeneratorImpl {
    VulkanGeneratorImpl(c10::DeviceIndex device_index = 0);
    ~VulkanGeneratorImpl() override = default;

    // Required overrides
    void set_current_seed(uint64_t seed) override;
    void set_offset(uint64_t offset) override;
    uint64_t get_offset() const override;
    uint64_t current_seed() const override;
    uint64_t seed() override;
    void set_state(const c10::TensorImpl& new_state) override;
    c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

    std::shared_ptr<VulkanGeneratorImpl> clone() const;

private:
    VulkanGeneratorImpl* clone_impl() const override;

    uint64_t seed_ = c10::default_rng_seed_val;
    uint64_t offset_ = 0;
};

at::Generator MakeVulkanGenerator(c10::DeviceIndex device_index);

} // namespace torch_vulkan
