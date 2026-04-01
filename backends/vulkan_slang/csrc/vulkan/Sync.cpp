#include "Sync.h"
#include <stdexcept>

namespace vulkan {

// ── Fence ────────────────────────────────────────────────────────
Fence::Fence(VkDevice device, bool signaled) : device_(device) {
    VkFenceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkResult result = vkCreateFence(device_, &ci, nullptr, &fence_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }
}

Fence::~Fence() {
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
    }
}

void Fence::wait(uint64_t timeout_ns) {
    VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_ns);
    if (result == VK_TIMEOUT) {
        throw std::runtime_error("Fence wait timed out");
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Fence wait failed");
    }
}

void Fence::reset() {
    vkResetFences(device_, 1, &fence_);
}

bool Fence::is_signaled() const {
    return vkGetFenceStatus(device_, fence_) == VK_SUCCESS;
}

// ── Event ────────────────────────────────────────────────────────
Event::Event(VkDevice device) : device_(device) {
    VkEventCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;

    VkResult result = vkCreateEvent(device_, &ci, nullptr, &event_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create event");
    }
}

Event::~Event() {
    if (event_ != VK_NULL_HANDLE) {
        vkDestroyEvent(device_, event_, nullptr);
    }
}

void Event::set() {
    vkSetEvent(device_, event_);
}

void Event::reset() {
    vkResetEvent(device_, event_);
}

bool Event::is_set() const {
    return vkGetEventStatus(device_, event_) == VK_EVENT_SET;
}

} // namespace vulkan
