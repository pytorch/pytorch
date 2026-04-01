// d3d12-texture.cpp
#include "d3d12-texture.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

TextureResourceImpl::TextureResourceImpl(const Desc& desc)
    : Parent(desc), m_defaultState(D3DUtil::getResourceState(desc.defaultState))
{
}

TextureResourceImpl::~TextureResourceImpl()
{
    if (sharedHandle.handleValue != 0)
    {
        CloseHandle((HANDLE)sharedHandle.handleValue);
    }
}

Result TextureResourceImpl::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = (uint64_t)m_resource.getResource();
    outHandle->api = InteropHandleAPI::D3D12;
    return SLANG_OK;
}

Result TextureResourceImpl::getSharedHandle(InteropHandle* outHandle)
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
    return SLANG_OK;
#endif
}

Result TextureResourceImpl::setDebugName(const char* name)
{
    Parent::setDebugName(name);
    m_resource.setDebugName(name);
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
