// metal-resource-views.h
#pragma once

#include "metal-base.h"
#include "metal-buffer.h"
#include "metal-device.h"
#include "metal-texture.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class ResourceViewImpl : public ResourceViewBase
{
public:
    enum class ViewType
    {
        Texture,
        Buffer,
        TexelBuffer,
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
    NS::SharedPtr<MTL::Texture> m_textureView;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class BufferResourceViewImpl : public ResourceViewImpl
{
public:
    BufferResourceViewImpl(DeviceImpl* device)
        : ResourceViewImpl(ViewType::Buffer, device)
    {
    }
    ~BufferResourceViewImpl();
    RefPtr<BufferResourceImpl> m_buffer;
    Offset m_offset;
    Size m_size;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class TexelBufferResourceViewImpl : public ResourceViewImpl
{
public:
    TexelBufferResourceViewImpl(DeviceImpl* device);
    ~TexelBufferResourceViewImpl();
    RefPtr<BufferResourceImpl> m_buffer;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

class AccelerationStructureImpl : public AccelerationStructureBase
{
public:
    RefPtr<BufferResourceImpl> m_buffer;
    RefPtr<DeviceImpl> m_device;

public:
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
    ~AccelerationStructureImpl();
};

} // namespace metal
} // namespace gfx
