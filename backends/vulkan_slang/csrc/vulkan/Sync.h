#pragma once

#include <vulkan/vulkan.h>

namespace vulkan {

class Fence {
public:
    Fence(VkDevice device, bool signaled = false);
    ~Fence();

    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;

    void wait(uint64_t timeout_ns = UINT64_MAX);
    void reset();
    bool is_signaled() const;

    VkFence handle() const { return fence_; }

private:
    VkDevice device_;
    VkFence fence_ = VK_NULL_HANDLE;
};

class Event {
public:
    Event(VkDevice device);
    ~Event();

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void set();
    void reset();
    bool is_set() const;

    VkEvent handle() const { return event_; }

private:
    VkDevice device_;
    VkEvent event_ = VK_NULL_HANDLE;
};

} // namespace vulkan
