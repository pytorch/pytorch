#pragma once

#include "CommandBuffer.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <unordered_set>

namespace vulkan {

class Stream {
public:
    Stream(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Submit a command buffer and wait
    void submit_and_wait(VkCommandBuffer cmd);

    // Submit a command buffer with fence
    VkFence submit(VkCommandBuffer cmd);

    // Wait for all pending work
    void synchronize();

    // Check if idle
    bool is_idle() const;

    // ── Deferred execution (command buffer batching) ──────────────
    // Get the active command buffer for recording dispatches.
    // Automatically begins a new command buffer if none is active.
    CommandBuffer& deferred_cmd();

    // Flush: end the active command buffer, submit, and wait.
    // No-op if no dispatches are pending.
    void flush();

    // Number of dispatches recorded in the current deferred command buffer.
    uint32_t pending_dispatches() const { return pending_dispatches_; }
    void inc_pending() { pending_dispatches_++; }

    // Track buffers used in current deferred batch for WAR hazard detection
    void track_buffer(VkBuffer buf) { pending_buffers_.insert(buf); }
    bool is_buffer_pending(VkBuffer buf) const { return pending_buffers_.count(buf) > 0; }

    VkQueue queue() const { return queue_; }
    CommandPool& command_pool() { return *cmd_pool_; }

private:
    VkDevice device_;
    VkQueue queue_;
    std::unique_ptr<CommandPool> cmd_pool_;
    VkFence fence_ = VK_NULL_HANDLE;

    // Deferred command buffer state
    std::unique_ptr<CommandBuffer> deferred_cmd_;
    uint32_t pending_dispatches_ = 0;
    std::unordered_set<VkBuffer> pending_buffers_;
};

} // namespace vulkan
