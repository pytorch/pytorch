// d3d12-fence.cpp
#include "d3d12-fence.h"

#include "d3d12-device.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

FenceImpl::~FenceImpl()
{
    if (m_waitEvent)
        CloseHandle(m_waitEvent);
}

HANDLE FenceImpl::getWaitEvent()
{
    if (m_waitEvent)
        return m_waitEvent;
    m_waitEvent = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    return m_waitEvent;
}

Result FenceImpl::init(DeviceImpl* device, const IFence::Desc& desc)
{
    SLANG_RETURN_ON_FAIL(device->m_device->CreateFence(
        desc.initialValue,
        desc.isShared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(m_fence.writeRef())));
    return SLANG_OK;
}

Result FenceImpl::getCurrentValue(uint64_t* outValue)
{
    *outValue = m_fence->GetCompletedValue();
    return SLANG_OK;
}

Result FenceImpl::setCurrentValue(uint64_t value)
{
    SLANG_RETURN_ON_FAIL(m_fence->Signal(value));
    return SLANG_OK;
}

Result FenceImpl::getSharedHandle(InteropHandle* outHandle)
{
#if !SLANG_WINDOWS_FAMILY
    return SLANG_E_NOT_IMPLEMENTED;
#else
    // Check if a shared handle already exists.
    if (sharedHandle.handleValue != 0)
    {
        *outHandle = sharedHandle;
        return SLANG_OK;
    }

    ComPtr<ID3D12Device> devicePtr;
    m_fence->GetDevice(IID_PPV_ARGS(devicePtr.writeRef()));
    SLANG_RETURN_ON_FAIL(devicePtr->CreateSharedHandle(
        m_fence,
        NULL,
        GENERIC_ALL,
        nullptr,
        (HANDLE*)&outHandle->handleValue));
    outHandle->api = InteropHandleAPI::D3D12;
    sharedHandle = *outHandle;
    return SLANG_OK;
#endif
}

Result FenceImpl::getNativeHandle(InteropHandle* outNativeHandle)
{
    outNativeHandle->api = gfx::InteropHandleAPI::D3D12;
    outNativeHandle->handleValue = (uint64_t)m_fence.get();
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
