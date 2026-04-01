#include "DeviceGuard.h"
#include "../vulkan/Context.h"

namespace torch_vulkan {

static thread_local c10::DeviceIndex current_device = 0;
static thread_local int64_t current_stream_id = 0;

c10::Device VulkanGuardImpl::exchangeDevice(c10::Device device) const {
    auto old = c10::Device(kDeviceType, current_device);
    current_device = device.index();
    vulkan::Context::instance().set_device(static_cast<uint32_t>(current_device));
    return old;
}

c10::Device VulkanGuardImpl::getDevice() const {
    return c10::Device(kDeviceType, current_device);
}

void VulkanGuardImpl::setDevice(c10::Device device) const {
    current_device = device.index();
    vulkan::Context::instance().set_device(static_cast<uint32_t>(current_device));
}

void VulkanGuardImpl::uncheckedSetDevice(c10::Device device) const noexcept {
    current_device = device.index();
}

c10::Stream VulkanGuardImpl::getStream(c10::Device device) const {
    return c10::Stream(c10::Stream::UNSAFE, device, current_stream_id);
}

c10::Stream VulkanGuardImpl::exchangeStream(c10::Stream stream) const noexcept {
    auto old = c10::Stream(c10::Stream::UNSAFE,
                            c10::Device(kDeviceType, current_device),
                            current_stream_id);
    current_stream_id = stream.id();
    return old;
}

c10::DeviceIndex VulkanGuardImpl::deviceCount() const noexcept {
    return static_cast<c10::DeviceIndex>(
        vulkan::Context::instance().device_count());
}

// ── Event support ──────────────────────────────────────────────────
// Single-stream backend: all work is serialized on one queue.
// Events are trivial — record marks done, query always returns true
// since by the time we check, the synchronous dispatch has completed.

void VulkanGuardImpl::record(
    void** event,
    const c10::Stream& /*stream*/,
    const c10::DeviceIndex device_index,
    const c10::EventFlag /*flag*/) const {
    if (*event == nullptr) {
        *event = new VulkanEvent();
    }
    auto* ve = static_cast<VulkanEvent*>(*event);
    ve->device_index = device_index;
    ve->recorded.store(true, std::memory_order_release);
}

void VulkanGuardImpl::block(
    void* /*event*/, const c10::Stream& /*stream*/) const {
    // Single-stream: no-op, all work is already ordered.
}

bool VulkanGuardImpl::queryEvent(void* /*event*/) const {
    // Single-stream synchronous backend: all dispatched work is complete
    // by the time control returns. Always return true.
    return true;
}

void VulkanGuardImpl::destroyEvent(
    void* event, const c10::DeviceIndex /*device_index*/) const noexcept {
    if (event != nullptr) {
        // Only delete events we allocated (check magic)
        auto* ve = static_cast<VulkanEvent*>(event);
        if (ve->is_valid()) {
            delete ve;
        }
    }
}

void VulkanGuardImpl::synchronizeEvent(void* /*event*/) const {
    // Single-stream: no-op.
}

// ── Stream support ─────────────────────────────────────────────────

bool VulkanGuardImpl::queryStream(const c10::Stream& /*stream*/) const {
    // Synchronous backend: stream is always idle after dispatch returns.
    return true;
}

void VulkanGuardImpl::synchronizeStream(const c10::Stream& /*stream*/) const {
    // Synchronous backend: nothing to wait for.
}

void VulkanGuardImpl::synchronizeDevice(
    const c10::DeviceIndex /*device_index*/) const {
    // Synchronous backend: nothing to wait for.
}

// Register the device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

} // namespace torch_vulkan
