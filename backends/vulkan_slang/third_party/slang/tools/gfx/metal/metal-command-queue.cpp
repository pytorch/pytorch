// metal-command-queue.cpp
#include "metal-command-queue.h"

#include "metal-command-buffer.h"
#include "metal-fence.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

ICommandQueue* CommandQueueImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandQueue)
        return static_cast<ICommandQueue*>(this);
    return nullptr;
}

CommandQueueImpl::~CommandQueueImpl() {}

void CommandQueueImpl::init(DeviceImpl* device, NS::SharedPtr<MTL::CommandQueue> commandQueue)
{
    m_device = device;
    m_commandQueue = commandQueue;
}

void CommandQueueImpl::waitOnHost()
{
    // TODO implement
}

Result CommandQueueImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Metal;
    outHandle->handleValue = reinterpret_cast<intptr_t>(m_commandQueue.get());
    return SLANG_OK;
}

const CommandQueueImpl::Desc& CommandQueueImpl::getDesc()
{
    return m_desc;
}

Result CommandQueueImpl::waitForFenceValuesOnDevice(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* waitValues)
{
    for (GfxCount i = 0; i < fenceCount; ++i)
    {
        FenceWaitInfo waitInfo;
        waitInfo.fence = static_cast<FenceImpl*>(fences[i]);
        waitInfo.waitValue = waitValues[i];
        m_pendingWaitFences.add(waitInfo);
    }
    return SLANG_OK;
}

void CommandQueueImpl::queueSubmitImpl(
    uint32_t count,
    ICommandBuffer* const* commandBuffers,
    IFence* fence,
    uint64_t valueToSignal)
{
    // If there are any pending wait fences, encode them to a new command buffer.
    // Metal ensures that command buffers are executed in the order they are committed.
    if (m_pendingWaitFences.getCount() > 0)
    {
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        for (const auto& fenceInfo : m_pendingWaitFences)
        {
            commandBuffer->encodeWait(fenceInfo.fence->m_event.get(), fenceInfo.waitValue);
        }
        commandBuffer->commit();
        m_pendingWaitFences.clear();
    }

    for (uint32_t i = 0; i < count; ++i)
    {
        CommandBufferImpl* cmdBufImpl = static_cast<CommandBufferImpl*>(commandBuffers[i]);
        // If this is the last command buffer and a fence is provided, signal the fence.
        if (i == count - 1 && fence != nullptr)
        {
            cmdBufImpl->m_commandBuffer->encodeSignalEvent(
                static_cast<FenceImpl*>(fence)->m_event.get(),
                valueToSignal);
        }
        cmdBufImpl->m_commandBuffer->commit();
    }

    // If there are no command buffers to submit, but a fence is provided, signal the fence.
    if (count == 0 && fence != nullptr)
    {
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        commandBuffer->encodeSignalEvent(
            static_cast<FenceImpl*>(fence)->m_event.get(),
            valueToSignal);
        commandBuffer->commit();
    }
}

void CommandQueueImpl::executeCommandBuffers(
    GfxCount count,
    ICommandBuffer* const* commandBuffers,
    IFence* fence,
    uint64_t valueToSignal)
{
    AUTORELEASEPOOL

    if (count == 0 && fence == nullptr)
    {
        return;
    }
    queueSubmitImpl(count, commandBuffers, fence, valueToSignal);
}

} // namespace metal
} // namespace gfx
