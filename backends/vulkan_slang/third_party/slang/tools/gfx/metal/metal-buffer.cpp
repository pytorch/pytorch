// metal-buffer.cpp
#include "metal-buffer.h"

#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

BufferResourceImpl::BufferResourceImpl(const IBufferResource::Desc& desc, DeviceImpl* device)
    : Parent(desc), m_device(device)
{
}

BufferResourceImpl::~BufferResourceImpl() {}

DeviceAddress BufferResourceImpl::getDeviceAddress()
{
    return m_buffer->gpuAddress();
}

Result BufferResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<intptr_t>(m_buffer.get());
    return SLANG_OK;
}

Result BufferResourceImpl::getSharedHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_AVAILABLE;
}

Result BufferResourceImpl::map(MemoryRange* rangeToRead, void** outPointer)
{
    *outPointer = m_buffer->contents();
    return SLANG_OK;
}

Result BufferResourceImpl::unmap(MemoryRange* writtenRange)
{
    return SLANG_OK;
}

Result BufferResourceImpl::setDebugName(const char* name)
{
    Parent::setDebugName(name);
    m_buffer->addDebugMarker(MetalUtil::createString(name).get(), NS::Range(0, m_desc.sizeInBytes));
    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
