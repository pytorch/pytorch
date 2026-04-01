#include "Memory.h"

#include <cstring>
#include <stdexcept>

namespace vulkan {

// ── Pre-read callback ───────────────────────────────────────────
static PreReadCallback g_pre_read_callback = nullptr;
static IsBufferInFlightCallback g_is_buffer_in_flight_callback = nullptr;

void set_pre_read_callback(PreReadCallback cb) {
    g_pre_read_callback = cb;
}

void set_is_buffer_in_flight_callback(IsBufferInFlightCallback cb) {
    g_is_buffer_in_flight_callback = cb;
}

VulkanBuffer::VulkanBuffer(VmaAllocator allocator, VkDeviceSize size, BufferType type)
    : allocator_(allocator), size_(size) {

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    switch (type) {
        case BufferType::DeviceLocal:
            alloc_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
        case BufferType::HostVisible:
            alloc_ci.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case BufferType::Staging:
            alloc_ci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
    }

    VkResult result = vmaCreateBuffer(allocator_, &buf_ci, &alloc_ci,
                                       &buffer_, &allocation_, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("VMA: Failed to allocate buffer of size " +
                                 std::to_string(size));
    }
}

VulkanBuffer::~VulkanBuffer() {
    release();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      size_(other.size_),
      is_mapped_(other.is_mapped_),
      mapped_ptr_(other.mapped_ptr_) {
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.is_mapped_ = false;
    other.mapped_ptr_ = nullptr;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        release();
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        size_ = other.size_;
        is_mapped_ = other.is_mapped_;
        mapped_ptr_ = other.mapped_ptr_;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.size_ = 0;
        other.is_mapped_ = false;
        other.mapped_ptr_ = nullptr;
    }
    return *this;
}

void VulkanBuffer::release() {
    if (buffer_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        if (is_mapped_) {
            vmaUnmapMemory(allocator_, allocation_);
        }
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

void* VulkanBuffer::map() {
    if (!is_mapped_) {
        VkResult result = vmaMapMemory(allocator_, allocation_, &mapped_ptr_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("VMA: Failed to map buffer memory");
        }
        is_mapped_ = true;
    }
    return mapped_ptr_;
}

void VulkanBuffer::unmap() {
    if (is_mapped_) {
        vmaUnmapMemory(allocator_, allocation_);
        is_mapped_ = false;
        mapped_ptr_ = nullptr;
    }
}

void VulkanBuffer::write(const void* data, VkDeviceSize write_size, VkDeviceSize offset) {
    void* ptr = map();
    std::memcpy(static_cast<char*>(ptr) + offset, data, write_size);
    vmaFlushAllocation(allocator_, allocation_, offset, write_size);
    unmap();
}

void VulkanBuffer::read(void* data, VkDeviceSize read_size, VkDeviceSize offset) const {
    // Only flush if THIS buffer has pending GPU work.
    // Flushing all pending work for every read is expensive — buffers that
    // were just allocated or only written from CPU don't need a flush.
    if (g_pre_read_callback && g_is_buffer_in_flight_callback &&
        g_is_buffer_in_flight_callback(buffer_)) {
        g_pre_read_callback();
    }

    void* ptr = nullptr;
    vmaMapMemory(allocator_, allocation_, &ptr);
    vmaInvalidateAllocation(allocator_, allocation_, offset, read_size);
    std::memcpy(data, static_cast<const char*>(ptr) + offset, read_size);
    vmaUnmapMemory(allocator_, allocation_);
}

// ── Buffer-to-buffer copy ────────────────────────────────────────
void copy_buffer(VkDevice device, VkCommandPool cmd_pool, VkQueue queue,
                 VkBuffer src, VkBuffer dst, VkDeviceSize size,
                 VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
}

} // namespace vulkan
