#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <atomic>

namespace torch_vulkan {

// Simple event for single-stream backend: just tracks whether all prior
// work has completed. Since we have one compute queue, recording an event
// means "everything submitted before this point".
struct VulkanEvent {
    static constexpr uint64_t kMagic = 0x564B4556454E5400ULL; // "VKEVNT\0\0"
    uint64_t magic{kMagic};
    std::atomic<bool> recorded{false};
    c10::DeviceIndex device_index{0};

    bool is_valid() const { return magic == kMagic; }
};

struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

    VulkanGuardImpl() = default;
    explicit VulkanGuardImpl(c10::DeviceType) {}

    c10::DeviceType type() const override { return kDeviceType; }

    c10::Device exchangeDevice(c10::Device device) const override;
    c10::Device getDevice() const override;
    void setDevice(c10::Device device) const override;
    void uncheckedSetDevice(c10::Device device) const noexcept override;

    c10::Stream getStream(c10::Device device) const override;
    c10::Stream exchangeStream(c10::Stream stream) const noexcept override;
    c10::DeviceIndex deviceCount() const noexcept override;

    // Event support (required for autograd engine)
    void record(
        void** event,
        const c10::Stream& stream,
        const c10::DeviceIndex device_index,
        const c10::EventFlag flag) const override;
    void block(void* event, const c10::Stream& stream) const override;
    bool queryEvent(void* event) const override;
    void destroyEvent(void* event, const c10::DeviceIndex device_index)
        const noexcept override;
    void synchronizeEvent(void* event) const override;

    // Stream support (required for autograd engine)
    bool queryStream(const c10::Stream& stream) const override;
    void synchronizeStream(const c10::Stream& stream) const override;
    void synchronizeDevice(const c10::DeviceIndex device_index) const override;
};

} // namespace torch_vulkan
