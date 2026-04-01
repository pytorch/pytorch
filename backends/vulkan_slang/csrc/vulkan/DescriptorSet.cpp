#include "DescriptorSet.h"
#include <stdexcept>

namespace vulkan {

DescriptorPool::DescriptorPool(VkDevice device, uint32_t max_sets)
    : device_(device) {

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = max_sets * 16; // Up to 16 bindings per set (cat_n uses 9)

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &pool_size;

    VkResult result = vkCreateDescriptorPool(device_, &ci, nullptr, &pool_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

DescriptorPool::~DescriptorPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
}

VkDescriptorSet DescriptorPool::allocate(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;

    VkDescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(device_, &ai, &set);
    if (result != VK_SUCCESS) {
        // Pool exhausted — flush pending GPU work then reset
        if (pre_reset_cb_) pre_reset_cb_();
        reset();
        result = vkAllocateDescriptorSets(device_, &ai, &set);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }
    }
    return set;
}

void DescriptorPool::reset() {
    vkResetDescriptorPool(device_, pool_, 0);
}

void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const VkBuffer* buffers,
                  const VkDeviceSize* sizes,
                  uint32_t count) {
    // Stack-allocated arrays — max 32 bindings (sgd_batch15 uses 30: 15 params + 15 grads)
    constexpr uint32_t MAX_BINDINGS = 32;
    VkDescriptorBufferInfo buf_infos[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];

    for (uint32_t i = 0; i < count; i++) {
        buf_infos[i] = {};
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = sizes[i];

        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }

    vkUpdateDescriptorSets(device, count, writes, 0, nullptr);
}

void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const std::vector<VkBuffer>& buffers,
                  const std::vector<VkDeviceSize>& sizes) {
    bind_buffers(device, set, buffers.data(), sizes.data(),
                 static_cast<uint32_t>(buffers.size()));
}

} // namespace vulkan
