// metal-resource-views.cpp
#include "metal-resource-views.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

TextureResourceViewImpl::~TextureResourceViewImpl() {}

Result TextureResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<uintptr_t>(m_textureView.get());
    return SLANG_OK;
}

BufferResourceViewImpl::~BufferResourceViewImpl() {}

Result BufferResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<uintptr_t>(m_buffer->m_buffer.get());
    return SLANG_OK;
}

TexelBufferResourceViewImpl::TexelBufferResourceViewImpl(DeviceImpl* device)
    : ResourceViewImpl(ViewType::TexelBuffer, device)
{
}

TexelBufferResourceViewImpl::~TexelBufferResourceViewImpl() {}

Result TexelBufferResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

DeviceAddress AccelerationStructureImpl::getDeviceAddress()
{
    return 0;
}

Result AccelerationStructureImpl::getNativeHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

AccelerationStructureImpl::~AccelerationStructureImpl() {}

} // namespace metal
} // namespace gfx
