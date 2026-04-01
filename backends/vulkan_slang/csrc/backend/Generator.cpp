#include "Generator.h"
#include "../ops/ops.h"

#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/Utils.h>
#include <c10/util/irange.h>

#include <random>

namespace torch_vulkan {

VulkanGeneratorImpl::VulkanGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl(
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {
    seed_ = c10::default_rng_seed_val;
    offset_ = 0;
}

void VulkanGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    offset_ = 0;
    // Propagate to the global RNG state used by shader dispatch
    ops::vulkan_manual_seed(seed);
}

void VulkanGeneratorImpl::set_offset(uint64_t offset) {
    offset_ = offset;
}

uint64_t VulkanGeneratorImpl::get_offset() const {
    return offset_;
}

uint64_t VulkanGeneratorImpl::current_seed() const {
    return seed_;
}

uint64_t VulkanGeneratorImpl::seed() {
    // Generate a random seed
    std::random_device rd;
    uint64_t new_seed = (uint64_t(rd()) << 32) | rd();
    set_current_seed(new_seed);
    return new_seed;
}

void VulkanGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
    // State is stored as a uint8 tensor containing [seed (8 bytes), offset (8 bytes)]
    auto data = static_cast<const uint8_t*>(new_state.data());
    TORCH_CHECK(new_state.numel() >= 16, "VulkanGenerator state must be at least 16 bytes");
    uint64_t new_seed, new_offset;
    std::memcpy(&new_seed, data, sizeof(uint64_t));
    std::memcpy(&new_offset, data + sizeof(uint64_t), sizeof(uint64_t));
    seed_ = new_seed;
    offset_ = new_offset;
    ops::vulkan_manual_seed(seed_);
}

c10::intrusive_ptr<c10::TensorImpl> VulkanGeneratorImpl::get_state() const {
    // Return state as a CPU byte tensor containing [seed, offset]
    auto state = at::detail::empty_cpu({16}, at::ScalarType::Byte, std::nullopt,
                                        std::nullopt, std::nullopt, std::nullopt);
    auto* data = state.data_ptr<uint8_t>();
    std::memcpy(data, &seed_, sizeof(uint64_t));
    std::memcpy(data + sizeof(uint64_t), &offset_, sizeof(uint64_t));
    return state.getIntrusivePtr();
}

std::shared_ptr<VulkanGeneratorImpl> VulkanGeneratorImpl::clone() const {
    return std::shared_ptr<VulkanGeneratorImpl>(clone_impl());
}

VulkanGeneratorImpl* VulkanGeneratorImpl::clone_impl() const {
    auto gen = new VulkanGeneratorImpl(this->device().index());
    gen->seed_ = this->seed_;
    gen->offset_ = this->offset_;
    return gen;
}

at::Generator MakeVulkanGenerator(c10::DeviceIndex device_index) {
    return at::make_generator<VulkanGeneratorImpl>(device_index);
}

} // namespace torch_vulkan
