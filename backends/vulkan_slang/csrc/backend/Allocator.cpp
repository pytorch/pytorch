#include "Allocator.h"
#include "../ops/dispatch.h"

#include <c10/core/Device.h>
#include <cstring>

namespace torch_vulkan {

VulkanAllocator& VulkanAllocator::instance() {
    static VulkanAllocator alloc;
    return alloc;
}

// Round up to next power-of-2 size class (min 512 bytes)
// This reduces fragmentation by grouping allocations into size buckets.
VkDeviceSize VulkanAllocator::round_size(VkDeviceSize nbytes) {
    if (nbytes <= 512) return 512;
    // Round to next power of 2
    VkDeviceSize size = 1;
    while (size < nbytes) size <<= 1;
    return size;
}

std::unique_ptr<vulkan::VulkanBuffer> VulkanAllocator::try_get_cached(VkDeviceSize size) {
    VkDeviceSize size_class = round_size(size);
    auto it = pool_.find(size_class);
    if (it != pool_.end() && !it->second.empty()) {
        auto buffer = std::move(it->second.back());
        it->second.pop_back();
        if (it->second.empty()) pool_.erase(it);
        cached_bytes_ -= size_class;
        return buffer;
    }
    return nullptr;
}

void VulkanAllocator::return_to_cache(std::unique_ptr<vulkan::VulkanBuffer> buffer) {
    VkDeviceSize size_class = round_size(buffer->size());

    // If cache is too large, don't cache this buffer — just drop it
    if (cached_bytes_ + size_class > kMaxCachedBytes) {
        buffer.reset();  // Free immediately
        return;
    }

    cached_bytes_ += size_class;
    pool_[size_class].push_back(std::move(buffer));
}

c10::DataPtr VulkanAllocator::allocate(size_t nbytes) {
    if (nbytes == 0) {
        return c10::DataPtr(nullptr, nullptr, &deleter,
                            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    auto& ctx = vulkan::Context::instance();
    VkDeviceSize alloc_size = round_size(static_cast<VkDeviceSize>(nbytes));

    std::unique_ptr<vulkan::VulkanBuffer> buffer;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer = try_get_cached(alloc_size);
    }

    // No WAR check needed: freed buffers are quarantined in pending_recycle_
    // until the command buffer they were referenced by completes (drain_pending_recycle).
    // Buffers in pool_ are guaranteed safe to reuse.

    if (!buffer) {
        buffer = std::make_unique<vulkan::VulkanBuffer>(
            ctx.allocator(), alloc_size,
            vulkan::BufferType::HostVisible);
    }

    // Use an opaque ID as the "data pointer"
    void* opaque = reinterpret_cast<void*>(next_id_++);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_[opaque] = std::move(buffer);
    }

    return c10::DataPtr(
        opaque, opaque, &deleter,
        c10::Device(c10::DeviceType::PrivateUse1,
                    static_cast<c10::DeviceIndex>(ctx.current_device())));
}

c10::DeleterFnPtr VulkanAllocator::raw_deleter() const {
    return &deleter;
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    std::memcpy(dest, src, count);
}

void VulkanAllocator::deleter(void* ptr) {
    if (!ptr) return;
    auto& alloc = VulkanAllocator::instance();
    std::lock_guard<std::mutex> lock(alloc.mutex_);

    // After shutdown, Vulkan device is destroyed — skip cleanup
    if (alloc.shutdown_) return;

    auto it = alloc.buffers_.find(ptr);
    if (it != alloc.buffers_.end()) {
        // Quarantine buffer: don't put back in pool yet.
        // It may still be referenced by the pending command buffer.
        // drain_pending_recycle() moves these to pool after flush.
        alloc.pending_recycle_.push_back(std::move(it->second));
        alloc.buffers_.erase(it);
    }
}

void VulkanAllocator::drain_pending_recycle() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& buffer : pending_recycle_) {
        return_to_cache(std::move(buffer));
    }
    pending_recycle_.clear();
}

vulkan::VulkanBuffer* VulkanAllocator::get_buffer(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    if (it == buffers_.end()) return nullptr;
    return it->second.get();
}

void VulkanAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.clear();
    pending_recycle_.clear();
    cached_bytes_ = 0;
}

void VulkanAllocator::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    buffers_.clear();
    pool_.clear();
    pending_recycle_.clear();
    cached_bytes_ = 0;
}

size_t VulkanAllocator::cached_bytes() const {
    return cached_bytes_;
}

} // namespace torch_vulkan
