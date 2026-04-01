// d3d12-transient-heap.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class TransientResourceHeapImpl
    : public TransientResourceHeapBaseImpl<DeviceImpl, BufferResourceImpl>,
      public ITransientResourceHeapD3D12
{
private:
    typedef TransientResourceHeapBaseImpl<DeviceImpl, BufferResourceImpl> Super;

public:
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    List<ComPtr<ID3D12GraphicsCommandList>> m_d3dCommandListPool;
    List<RefPtr<RefObject>> m_commandBufferPool;
    uint32_t m_commandListAllocId = 0;
    // Wait values for each command queue.
    struct QueueWaitInfo
    {
        uint64_t waitValue;
        HANDLE fenceEvent;
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12Fence> fence = nullptr;
    };
    ShortList<QueueWaitInfo, 4> m_waitInfos;
    ShortList<HANDLE, 4> m_waitHandles;

    QueueWaitInfo& getQueueWaitInfo(uint32_t queueIndex);
    // During command submission, we need all the descriptor tables that get
    // used to come from a single heap (for each descriptor heap type).
    //
    // We will thus keep a single heap of each type that we hope will hold
    // all the descriptors that actually get needed in a frame.
    ShortList<D3D12DescriptorHeap, 4> m_viewHeaps;    // Cbv, Srv, Uav
    ShortList<D3D12DescriptorHeap, 4> m_samplerHeaps; // Heap for samplers
    int32_t m_currentViewHeapIndex = -1;
    int32_t m_currentSamplerHeapIndex = -1;
    bool m_canResize = false;

    uint32_t m_viewHeapSize;
    uint32_t m_samplerHeapSize;

    D3D12DescriptorHeap& getCurrentViewHeap();
    D3D12DescriptorHeap& getCurrentSamplerHeap();

    D3D12LinearExpandingDescriptorHeap m_stagingCpuViewHeap;
    D3D12LinearExpandingDescriptorHeap m_stagingCpuSamplerHeap;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) override;

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return Super::addRef(); }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return Super::release(); }

    virtual SLANG_NO_THROW Result SLANG_MCALL allocateTransientDescriptorTable(
        DescriptorType type,
        GfxCount count,
        Offset& outDescriptorOffset,
        void** outD3DDescriptorHeapHandle) override;

    ~TransientResourceHeapImpl();

    bool canResize() { return m_canResize; }

    Result init(
        const ITransientResourceHeap::Desc& desc,
        DeviceImpl* device,
        uint32_t viewHeapSize,
        uint32_t samplerHeapSize);

    Result allocateNewViewDescriptorHeap(DeviceImpl* device);

    Result allocateNewSamplerDescriptorHeap(DeviceImpl* device);

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandBuffer(ICommandBuffer** outCommandBuffer) override;

    Result synchronize();

    virtual SLANG_NO_THROW Result SLANG_MCALL synchronizeAndReset() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL finish() override;
};

} // namespace d3d12
} // namespace gfx
