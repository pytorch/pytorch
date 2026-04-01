#pragma once

#include <vulkan/vulkan.h>
#include <vector>

namespace vulkan {

class DescriptorPool {
public:
    DescriptorPool(VkDevice device, uint32_t max_sets = 4096);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    VkDescriptorSet allocate(VkDescriptorSetLayout layout);
    void reset();

    // Set callback invoked before pool reset (to flush pending GPU work)
    using PreResetCallback = void(*)();
    void set_pre_reset_callback(PreResetCallback cb) { pre_reset_cb_ = cb; }

    VkDescriptorPool pool() const { return pool_; }

private:
    VkDevice device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    PreResetCallback pre_reset_cb_ = nullptr;
};

// Bind storage buffers to a descriptor set
void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const std::vector<VkBuffer>& buffers,
                  const std::vector<VkDeviceSize>& sizes);

// Stack-allocated version to avoid heap allocation per dispatch
void bind_buffers(VkDevice device,
                  VkDescriptorSet set,
                  const VkBuffer* buffers,
                  const VkDeviceSize* sizes,
                  uint32_t count);

} // namespace vulkan
