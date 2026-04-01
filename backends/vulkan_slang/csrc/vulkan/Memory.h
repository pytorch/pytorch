#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstddef>

namespace vulkan {

// Global pre-read callback: flushing pending GPU work before host reads.
// Set by the ops layer to inject flush_stream() without circular dependency.
using PreReadCallback = void(*)();
void set_pre_read_callback(PreReadCallback cb);

// Callback to check if a specific buffer is in a pending command buffer.
// Used to avoid unnecessary flushes when reading buffers that have no pending GPU work.
using IsBufferInFlightCallback = bool(*)(VkBuffer);
void set_is_buffer_in_flight_callback(IsBufferInFlightCallback cb);

// Buffer usage hints
enum class BufferType {
    DeviceLocal,     // GPU-only, fastest for compute
    HostVisible,     // CPU-visible, coherent, for small buffers / readback
    Staging,         // CPU-visible, used for transfers
};

class VulkanBuffer {
public:
    VulkanBuffer() = default;
    VulkanBuffer(VmaAllocator allocator, VkDeviceSize size, BufferType type);
    ~VulkanBuffer();

    // Move only
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    VmaAllocation allocation() const { return allocation_; }

    // Map/unmap for host-visible buffers
    void* map();
    void unmap();

    // Write data to buffer (must be host-visible or staging)
    void write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    // Read data from buffer (must be host-visible or staging)
    void read(void* data, VkDeviceSize size, VkDeviceSize offset = 0) const;

    bool is_valid() const { return buffer_ != VK_NULL_HANDLE; }

private:
    void release();

    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    bool is_mapped_ = false;
    void* mapped_ptr_ = nullptr;
};

// Copy between buffers using a command buffer
void copy_buffer(
    VkDevice device,
    VkCommandPool cmd_pool,
    VkQueue queue,
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize src_offset = 0,
    VkDeviceSize dst_offset = 0);

} // namespace vulkan
