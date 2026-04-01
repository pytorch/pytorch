// d3d12-buffer.cpp
#include "d3d12-buffer.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

BufferResourceImpl::BufferResourceImpl(const Desc& desc)
    : Parent(desc), m_defaultState(D3DUtil::getResourceState(desc.defaultState))
{
}

BufferResourceImpl::~BufferResourceImpl()
{
    if (sharedHandle.handleValue != 0)
    {
        CloseHandle((HANDLE)sharedHandle.handleValue);
    }
}

DeviceAddress BufferResourceImpl::getDeviceAddress()
{
    return (DeviceAddress)m_resource.getResource()->GetGPUVirtualAddress();
}

Result BufferResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = (uint64_t)m_resource.getResource();
    outHandle->api = InteropHandleAPI::D3D12;
    return SLANG_OK;
}

Result BufferResourceImpl::getSharedHandle(InteropHandle* outHandle)
{
#if !SLANG_WINDOWS_FAMILY
    return SLANG_E_NOT_IMPLEMENTED;
#else
    // Check if a shared handle already exists for this resource.
    if (sharedHandle.handleValue != 0)
    {
        *outHandle = sharedHandle;
        return SLANG_OK;
    }

    // If a shared handle doesn't exist, create one and store it.
    ComPtr<ID3D12Device> pDevice;
    auto pResource = m_resource.getResource();
    pResource->GetDevice(IID_PPV_ARGS(pDevice.writeRef()));
    SLANG_RETURN_ON_FAIL(pDevice->CreateSharedHandle(
        pResource,
        NULL,
        GENERIC_ALL,
        nullptr,
        (HANDLE*)&outHandle->handleValue));
    outHandle->api = InteropHandleAPI::D3D12;
    sharedHandle = *outHandle;
    return SLANG_OK;
#endif
}

Result BufferResourceImpl::map(MemoryRange* rangeToRead, void** outPointer)
{
    D3D12_RANGE range = {};
    if (rangeToRead)
    {
        range.Begin = (SIZE_T)rangeToRead->offset;
        range.End = (SIZE_T)(rangeToRead->offset + rangeToRead->size);
    }
    SLANG_RETURN_ON_FAIL(
        m_resource.getResource()->Map(0, rangeToRead ? &range : nullptr, outPointer));
    return SLANG_OK;
}

Result BufferResourceImpl::unmap(MemoryRange* writtenRange)
{
    D3D12_RANGE range = {};
    if (writtenRange)
    {
        range.Begin = (SIZE_T)writtenRange->offset;
        range.End = (SIZE_T)(writtenRange->offset + writtenRange->size);
    }
    m_resource.getResource()->Unmap(0, writtenRange ? &range : nullptr);
    return SLANG_OK;
}

Result BufferResourceImpl::setDebugName(const char* name)
{
    Parent::setDebugName(name);
    m_resource.setDebugName(name);
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
