// vk-resource-views.cpp
#include "vk-resource-views.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

TextureResourceViewImpl::~TextureResourceViewImpl()
{
    m_device->m_api.vkDestroyImageView(m_device->m_api.m_device, m_view, nullptr);
}

Result TextureResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = (uint64_t)(m_view);
    return SLANG_OK;
}

TexelBufferResourceViewImpl::TexelBufferResourceViewImpl(DeviceImpl* device)
    : ResourceViewImpl(ViewType::TexelBuffer, device)
{
}

TexelBufferResourceViewImpl::~TexelBufferResourceViewImpl()
{
    m_device->m_api.vkDestroyBufferView(m_device->m_api.m_device, m_view, nullptr);
}

Result TexelBufferResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = (uint64_t)(m_view);
    return SLANG_OK;
}

PlainBufferResourceViewImpl::PlainBufferResourceViewImpl(DeviceImpl* device)
    : ResourceViewImpl(ViewType::PlainBuffer, device)
{
}

Result PlainBufferResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    return m_buffer->getNativeResourceHandle(outHandle);
}

DeviceAddress AccelerationStructureImpl::getDeviceAddress()
{
    return m_buffer->getDeviceAddress() + m_offset;
}

Result AccelerationStructureImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = (uint64_t)(m_vkHandle);
    return SLANG_OK;
}

AccelerationStructureImpl::~AccelerationStructureImpl()
{
    if (m_device)
    {
        m_device->m_api.vkDestroyAccelerationStructureKHR(
            m_device->m_api.m_device,
            m_vkHandle,
            nullptr);
    }
}

} // namespace vk
} // namespace gfx
