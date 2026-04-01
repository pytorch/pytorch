#pragma once

#include <vulkan/vulkan.h>
#include <vector>

namespace vulkan {

class CommandPool {
public:
    CommandPool(VkDevice device, uint32_t queue_family);
    ~CommandPool();

    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    VkCommandBuffer allocate();
    void free(VkCommandBuffer cmd);
    void reset();

    VkCommandPool pool() const { return pool_; }

private:
    VkDevice device_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
};

class CommandBuffer {
public:
    CommandBuffer(VkDevice device, CommandPool& pool);
    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    void begin();
    void end();

    // Bind compute pipeline
    void bind_pipeline(VkPipeline pipeline);

    // Bind descriptor set
    void bind_descriptor_set(VkPipelineLayout layout, VkDescriptorSet set);

    // Push constants
    void push_constants(VkPipelineLayout layout, uint32_t size, const void* data);

    // Dispatch compute shader
    void dispatch(uint32_t group_count_x,
                  uint32_t group_count_y = 1,
                  uint32_t group_count_z = 1);

    // Pipeline barrier for compute
    void buffer_barrier(VkBuffer buffer, VkDeviceSize size,
                        VkAccessFlags src_access = VK_ACCESS_SHADER_WRITE_BIT,
                        VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT);

    // Memory barrier
    void memory_barrier(VkAccessFlags src_access, VkAccessFlags dst_access);

    VkCommandBuffer handle() const { return cmd_; }

private:
    VkDevice device_;
    CommandPool& pool_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool recording_ = false;
};

} // namespace vulkan
