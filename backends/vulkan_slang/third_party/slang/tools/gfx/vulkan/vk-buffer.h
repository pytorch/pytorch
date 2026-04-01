// vk-buffer.h
#pragma once

#include "vk-base.h"
#include "vk-device.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class VKBufferHandleRAII
{
public:
    /// Initialize a buffer with specified size, and memory props
    Result init(
        const VulkanApi& api,
        Size bufferSize,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags reqMemoryProperties,
        bool isShared = false,
        VkExternalMemoryHandleTypeFlagsKHR extMemHandleType = 0);

    /// Returns true if has been initialized
    bool isInitialized() const { return m_api != nullptr; }

    VKBufferHandleRAII()
        : m_api(nullptr)
    {
    }

    ~VKBufferHandleRAII()
    {
        if (m_api)
        {
            m_api->vkDestroyBuffer(m_api->m_device, m_buffer, nullptr);
            m_api->vkFreeMemory(m_api->m_device, m_memory, nullptr);
        }
    }

    VkBuffer m_buffer;
    VkDeviceMemory m_memory;
    const VulkanApi* m_api;
};

class BufferResourceImpl : public BufferResource
{
public:
    typedef BufferResource Parent;

    BufferResourceImpl(const IBufferResource::Desc& desc, DeviceImpl* renderer);

    ~BufferResourceImpl();

    RefPtr<DeviceImpl> m_renderer;
    VKBufferHandleRAII m_buffer;
    VKBufferHandleRAII m_uploadBuffer;

    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeResourceHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    map(MemoryRange* rangeToRead, void** outPointer) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL unmap(MemoryRange* writtenRange) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setDebugName(const char* name) override;
};

} // namespace vk
} // namespace gfx
