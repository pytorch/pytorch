#include "Stream.h"
#include <stdexcept>

namespace vulkan {

Stream::Stream(VkDevice device, VkQueue queue, uint32_t queue_family)
    : device_(device), queue_(queue) {
    cmd_pool_ = std::make_unique<CommandPool>(device, queue_family);

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkResult result = vkCreateFence(device_, &fence_ci, nullptr, &fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence for stream");
    }
}

Stream::~Stream() {
    // Flush any pending deferred work before destroying
    if (deferred_cmd_ && pending_dispatches_ > 0) {
        try { flush(); } catch (...) {}
    }
    deferred_cmd_.reset();
    if (fence_ != VK_NULL_HANDLE) {
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device_, fence_, nullptr);
    }
}

void Stream::submit_and_wait(VkCommandBuffer cmd) {
    vkResetFences(device_, 1, &fence_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkResult result = vkQueueSubmit(queue_, 1, &submit_info, fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }

    result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed waiting for fence");
    }
}

VkFence Stream::submit(VkCommandBuffer cmd) {
    vkResetFences(device_, 1, &fence_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkResult result = vkQueueSubmit(queue_, 1, &submit_info, fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
    return fence_;
}

void Stream::synchronize() {
    vkQueueWaitIdle(queue_);
}

bool Stream::is_idle() const {
    VkResult result = vkGetFenceStatus(device_, fence_);
    return result == VK_SUCCESS;
}

// ── Deferred execution ──────────────────────────────────────────

CommandBuffer& Stream::deferred_cmd() {
    if (!deferred_cmd_) {
        deferred_cmd_ = std::make_unique<CommandBuffer>(device_, *cmd_pool_);
        deferred_cmd_->begin();
    }
    return *deferred_cmd_;
}

void Stream::flush() {
    if (!deferred_cmd_ || pending_dispatches_ == 0) return;

    deferred_cmd_->end();
    submit_and_wait(deferred_cmd_->handle());

    // Release the command buffer and reset counter
    deferred_cmd_.reset();
    pending_dispatches_ = 0;
    pending_buffers_.clear();
}

} // namespace vulkan
