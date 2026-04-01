#pragma once

#include <c10/core/Allocator.h>
#include "../vulkan/Context.h"
#include "../vulkan/Memory.h"

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch_vulkan {

class VulkanAllocator final : public c10::Allocator {
public:
    static VulkanAllocator& instance();

    c10::DataPtr allocate(size_t nbytes) override;
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;

    // Get the VulkanBuffer backing a tensor's data pointer
    vulkan::VulkanBuffer* get_buffer(void* ptr);

    // Pool management
    void empty_cache();       // Release all cached buffers
    void release_all();       // Release ALL buffers (live + cached) — for shutdown
    size_t cached_bytes() const; // Total bytes in cache

private:
    VulkanAllocator() = default;

    static void deleter(void* ptr);

    // Round up to next size class for pooling
    static VkDeviceSize round_size(VkDeviceSize nbytes);

    // Try to get a buffer from the cache
    std::unique_ptr<vulkan::VulkanBuffer> try_get_cached(VkDeviceSize size);
    // Return a buffer to the cache
    void return_to_cache(std::unique_ptr<vulkan::VulkanBuffer> buffer);

    std::mutex mutex_;
    bool shutdown_ = false;
    std::unordered_map<void*, std::unique_ptr<vulkan::VulkanBuffer>> buffers_;
    uintptr_t next_id_ = 1;

    // Buffer pool: maps size_class -> list of available buffers
    std::map<VkDeviceSize, std::vector<std::unique_ptr<vulkan::VulkanBuffer>>> pool_;
    size_t cached_bytes_ = 0;
    static constexpr size_t kMaxCachedBytes = 256 * 1024 * 1024;  // 256 MB cache limit

    // Deferred recycling: buffers freed during a batch are quarantined here
    // until the batch is flushed. This avoids WAR hazard flushes.
    std::vector<std::unique_ptr<vulkan::VulkanBuffer>> pending_recycle_;

public:
    // Move quarantined buffers into the reuse pool. Called after flush.
    void drain_pending_recycle();
};

} // namespace torch_vulkan
