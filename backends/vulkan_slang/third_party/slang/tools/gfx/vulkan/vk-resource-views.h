// vk-resource-views.h
#pragma once

#include "vk-base.h"
#include "vk-buffer.h"
#include "vk-device.h"
#include "vk-texture.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class ResourceViewImpl : public ResourceViewBase
{
public:
    enum class ViewType
    {
        Texture,
        TexelBuffer,
        PlainBuffer,
    };

public:
    ResourceViewImpl(ViewType viewType, DeviceImpl* device)
        : m_type(viewType), m_device(device)
    {
    }
    ViewType m_type;
    RefPtr<DeviceImpl> m_device;
};

class TextureResourceViewImpl : public ResourceViewImpl
{
public:
    TextureResourceViewImpl(DeviceImpl* device)
        : ResourceViewImpl(ViewType::Texture, device)
    {
    }
    ~TextureResourceViewImpl();
    RefPtr<TextureResourceImpl> m_texture;
    VkImageView m_view;
    VkImageLayout m_layout;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class TexelBufferResourceViewImpl : public ResourceViewImpl
{
public:
    TexelBufferResourceViewImpl(DeviceImpl* device);
    ~TexelBufferResourceViewImpl();
    RefPtr<BufferResourceImpl> m_buffer;
    VkBufferView m_view;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class PlainBufferResourceViewImpl : public ResourceViewImpl
{
public:
    PlainBufferResourceViewImpl(DeviceImpl* device);
    RefPtr<BufferResourceImpl> m_buffer;
    VkDeviceSize offset;
    VkDeviceSize size;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class AccelerationStructureImpl : public AccelerationStructureBase
{
public:
    VkAccelerationStructureKHR m_vkHandle = VK_NULL_HANDLE;
    RefPtr<BufferResourceImpl> m_buffer;
    VkDeviceSize m_offset;
    VkDeviceSize m_size;
    RefPtr<DeviceImpl> m_device;

public:
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
    ~AccelerationStructureImpl();
};

} // namespace vk
} // namespace gfx
