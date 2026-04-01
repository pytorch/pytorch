// d3d12-command-queue.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class CommandQueueImpl : public ICommandQueue, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandQueue* getInterface(const Guid& guid);
    void breakStrongReferenceToDevice() { m_renderer.breakStrongReference(); }

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* handle) override;

public:
    BreakableReference<DeviceImpl> m_renderer;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_d3dQueue;
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE globalWaitHandle;
    Desc m_desc;
    uint32_t m_queueIndex = 0;

    Result init(DeviceImpl* device, uint32_t queueIndex);
    ~CommandQueueImpl();
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override;

    virtual SLANG_NO_THROW void SLANG_MCALL executeCommandBuffers(
        GfxCount count,
        ICommandBuffer* const* commandBuffers,
        IFence* fence,
        uint64_t valueToSignal) override;

    virtual SLANG_NO_THROW void SLANG_MCALL waitOnHost() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    waitForFenceValuesOnDevice(GfxCount fenceCount, IFence** fences, uint64_t* waitValues) override;
};

} // namespace d3d12
} // namespace gfx
