// d3d12-transient-heap.cpp
#include "d3d12-transient-heap.h"

#include "d3d12-buffer.h"
#include "d3d12-command-buffer.h"
#include "d3d12-device.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

Result TransientResourceHeapImpl::synchronize()
{
    WaitForMultipleObjects(
        (DWORD)m_waitHandles.getCount(),
        m_waitHandles.getArrayView().getBuffer(),
        TRUE,
        INFINITE);
    m_waitHandles.clear();
    return SLANG_OK;
}

Result TransientResourceHeapImpl::synchronizeAndReset()
{
    synchronize();

    m_currentViewHeapIndex = -1;
    m_currentSamplerHeapIndex = -1;
    allocateNewViewDescriptorHeap(m_device);
    allocateNewSamplerDescriptorHeap(m_device);
    m_stagingCpuSamplerHeap.freeAll();
    m_stagingCpuViewHeap.freeAll();
    m_commandListAllocId = 0;
    SLANG_RETURN_ON_FAIL(m_commandAllocator->Reset());
    Super::reset();
    return SLANG_OK;
}

Result TransientResourceHeapImpl::finish()
{
    for (auto& waitInfo : m_waitInfos)
    {
        if (waitInfo.waitValue == 0)
            continue;
        if (waitInfo.fence)
        {
            waitInfo.queue->Signal(waitInfo.fence, waitInfo.waitValue);
            waitInfo.fence->SetEventOnCompletion(waitInfo.waitValue, waitInfo.fenceEvent);
            m_waitHandles.add(waitInfo.fenceEvent);
        }
    }
    return SLANG_OK;
}

TransientResourceHeapImpl::QueueWaitInfo& TransientResourceHeapImpl::getQueueWaitInfo(
    uint32_t queueIndex)
{
    if (queueIndex < (uint32_t)m_waitInfos.getCount())
    {
        return m_waitInfos[queueIndex];
    }
    auto oldCount = m_waitInfos.getCount();
    m_waitInfos.setCount(queueIndex + 1);
    for (auto i = oldCount; i < m_waitInfos.getCount(); i++)
    {
        m_waitInfos[i].waitValue = 0;
        m_waitInfos[i].fenceEvent = CreateEventEx(nullptr, FALSE, 0, EVENT_ALL_ACCESS);
    }
    return m_waitInfos[queueIndex];
}

D3D12DescriptorHeap& TransientResourceHeapImpl::getCurrentViewHeap()
{
    return m_viewHeaps[m_currentViewHeapIndex];
}

D3D12DescriptorHeap& TransientResourceHeapImpl::getCurrentSamplerHeap()
{
    return m_samplerHeaps[m_currentSamplerHeapIndex];
}

Result TransientResourceHeapImpl::queryInterface(SlangUUID const& uuid, void** outObject)
{
    if (uuid == GfxGUID::IID_ITransientResourceHeapD3D12)
    {
        *outObject = static_cast<ITransientResourceHeapD3D12*>(this);
        addRef();
        return SLANG_OK;
    }
    return Super::queryInterface(uuid, outObject);
}

Result TransientResourceHeapImpl::allocateTransientDescriptorTable(
    DescriptorType type,
    GfxCount count,
    Offset& outDescriptorOffset,
    void** outD3DDescriptorHeapHandle)
{
    auto& heap =
        (type == DescriptorType::ResourceView) ? getCurrentViewHeap() : getCurrentSamplerHeap();
    int allocResult = heap.allocate((int)count);
    if (allocResult == -1)
    {
        return SLANG_E_OUT_OF_MEMORY;
    }
    outDescriptorOffset = (Offset)allocResult;
    *outD3DDescriptorHeapHandle = heap.getHeap();
    return SLANG_OK;
}

TransientResourceHeapImpl::~TransientResourceHeapImpl()
{
    synchronize();
    for (auto& waitInfo : m_waitInfos)
        CloseHandle(waitInfo.fenceEvent);
}

Result TransientResourceHeapImpl::init(
    const ITransientResourceHeap::Desc& desc,
    DeviceImpl* device,
    uint32_t viewHeapSize,
    uint32_t samplerHeapSize)
{
    Super::init(desc, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, device);
    m_canResize = (desc.flags & ITransientResourceHeap::Flags::AllowResizing) != 0;
    m_viewHeapSize = viewHeapSize;
    m_samplerHeapSize = samplerHeapSize;

    m_stagingCpuViewHeap.init(
        device->m_device,
        1000000,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    m_stagingCpuSamplerHeap.init(
        device->m_device,
        1000000,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE);

    auto d3dDevice = device->m_device;
    SLANG_RETURN_ON_FAIL(d3dDevice->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(m_commandAllocator.writeRef())));

    allocateNewViewDescriptorHeap(device);
    allocateNewSamplerDescriptorHeap(device);

    return SLANG_OK;
}

Result TransientResourceHeapImpl::allocateNewViewDescriptorHeap(DeviceImpl* device)
{
    auto nextHeapIndex = m_currentViewHeapIndex + 1;
    if (nextHeapIndex < m_viewHeaps.getCount())
    {
        m_viewHeaps[nextHeapIndex].deallocateAll();
        m_currentViewHeapIndex = nextHeapIndex;
        return SLANG_OK;
    }
    auto d3dDevice = device->m_device;
    D3D12DescriptorHeap viewHeap;
    SLANG_RETURN_ON_FAIL(viewHeap.init(
        d3dDevice,
        m_viewHeapSize,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));
    m_currentViewHeapIndex = (int32_t)m_viewHeaps.getCount();
    m_viewHeaps.add(_Move(viewHeap));
    return SLANG_OK;
}

Result TransientResourceHeapImpl::allocateNewSamplerDescriptorHeap(DeviceImpl* device)
{
    auto nextHeapIndex = m_currentSamplerHeapIndex + 1;
    if (nextHeapIndex < m_samplerHeaps.getCount())
    {
        m_samplerHeaps[nextHeapIndex].deallocateAll();
        m_currentSamplerHeapIndex = nextHeapIndex;
        return SLANG_OK;
    }
    auto d3dDevice = device->m_device;
    D3D12DescriptorHeap samplerHeap;
    SLANG_RETURN_ON_FAIL(samplerHeap.init(
        d3dDevice,
        m_samplerHeapSize,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));
    m_currentSamplerHeapIndex = (int32_t)m_samplerHeaps.getCount();
    m_samplerHeaps.add(_Move(samplerHeap));
    return SLANG_OK;
}

Result TransientResourceHeapImpl::createCommandBuffer(ICommandBuffer** outCmdBuffer)
{
    if ((Index)m_commandListAllocId < m_commandBufferPool.getCount())
    {
        auto result =
            static_cast<CommandBufferImpl*>(m_commandBufferPool[m_commandListAllocId].Ptr());
        m_d3dCommandListPool[m_commandListAllocId]->Reset(m_commandAllocator, nullptr);
        result->reinit();
        ++m_commandListAllocId;
        returnComPtr(outCmdBuffer, result);
        return SLANG_OK;
    }
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    SLANG_RETURN_ON_FAIL(m_device->m_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocator,
        nullptr,
        IID_PPV_ARGS(cmdList.writeRef())));

    m_d3dCommandListPool.add(cmdList);
    RefPtr<CommandBufferImpl> cmdBuffer = new CommandBufferImpl();
    cmdBuffer->init(m_device, cmdList, this);
    m_commandBufferPool.add(cmdBuffer);
    ++m_commandListAllocId;
    returnComPtr(outCmdBuffer, cmdBuffer);
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
