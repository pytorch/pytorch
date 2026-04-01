// d3d12-command-queue.cpp
#include "d3d12-command-queue.h"

#include "d3d12-command-buffer.h"
#include "d3d12-device.h"
#include "d3d12-fence.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

Result CommandQueueImpl::init(DeviceImpl* device, uint32_t queueIndex)
{
    m_queueIndex = queueIndex;
    m_renderer = device;
    m_device = device->m_device;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    SLANG_RETURN_ON_FAIL(
        m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_d3dQueue.writeRef())));
    SLANG_RETURN_ON_FAIL(
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.writeRef())));
    globalWaitHandle = CreateEventEx(
        nullptr,
        nullptr,
        CREATE_EVENT_INITIAL_SET | CREATE_EVENT_MANUAL_RESET,
        EVENT_ALL_ACCESS);
    return SLANG_OK;
}

CommandQueueImpl::~CommandQueueImpl()
{
    waitOnHost();
    CloseHandle(globalWaitHandle);
    m_renderer->m_queueIndexAllocator.free((int)m_queueIndex, 1);
}

void CommandQueueImpl::executeCommandBuffers(
    GfxCount count,
    ICommandBuffer* const* commandBuffers,
    IFence* fence,
    uint64_t valueToSignal)
{
    ShortList<ID3D12CommandList*> commandLists;
    for (GfxCount i = 0; i < count; i++)
    {
        auto cmdImpl = static_cast<CommandBufferImpl*>(commandBuffers[i]);
        commandLists.add(cmdImpl->m_cmdList);
    }
    if (count > 0)
    {
        m_d3dQueue->ExecuteCommandLists((UINT)count, commandLists.getArrayView().getBuffer());

        m_fenceValue++;

        for (GfxCount i = 0; i < count; i++)
        {
            if (i > 0 && commandBuffers[i] == commandBuffers[i - 1])
                continue;
            auto cmdImpl = static_cast<CommandBufferImpl*>(commandBuffers[i]);
            auto transientHeap = cmdImpl->m_transientHeap;
            auto& waitInfo = transientHeap->getQueueWaitInfo(m_queueIndex);
            waitInfo.waitValue = m_fenceValue;
            waitInfo.fence = m_fence;
            waitInfo.queue = m_d3dQueue;
        }
    }

    if (fence)
    {
        auto fenceImpl = static_cast<FenceImpl*>(fence);
        m_d3dQueue->Signal(fenceImpl->m_fence.get(), valueToSignal);
    }
}

void CommandQueueImpl::waitOnHost()
{
    m_fenceValue++;
    m_d3dQueue->Signal(m_fence, m_fenceValue);
    ResetEvent(globalWaitHandle);
    m_fence->SetEventOnCompletion(m_fenceValue, globalWaitHandle);
    WaitForSingleObject(globalWaitHandle, INFINITE);
}

Result CommandQueueImpl::waitForFenceValuesOnDevice(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* waitValues)
{
    for (GfxCount i = 0; i < fenceCount; ++i)
    {
        auto fenceImpl = static_cast<FenceImpl*>(fences[i]);
        m_d3dQueue->Wait(fenceImpl->m_fence.get(), waitValues[i]);
    }
    return SLANG_OK;
}

const CommandQueueImpl::Desc& CommandQueueImpl::getDesc()
{
    return m_desc;
}

ICommandQueue* CommandQueueImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandQueue)
        return static_cast<ICommandQueue*>(this);
    return nullptr;
}

Result CommandQueueImpl::getNativeHandle(InteropHandle* handle)
{
    handle->api = InteropHandleAPI::D3D12;
    handle->handleValue = (uint64_t)m_d3dQueue.get();
    return SLANG_OK;
}

} // namespace d3d12
} // namespace gfx
