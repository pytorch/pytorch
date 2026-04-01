// debug-command-queue.cpp
#include "debug-command-queue.h"

#include "debug-command-buffer.h"
#include "debug-fence.h"
#include "debug-helper-functions.h"
#include "debug-transient-heap.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

const ICommandQueue::Desc& DebugCommandQueue::getDesc()
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDesc();
}

void DebugCommandQueue::executeCommandBuffers(
    GfxCount count,
    ICommandBuffer* const* commandBuffers,
    IFence* fence,
    uint64_t valueToSignal)
{
    SLANG_GFX_API_FUNC;
    List<ICommandBuffer*> innerCommandBuffers;
    for (GfxIndex i = 0; i < count; i++)
    {
        auto cmdBufferIn = commandBuffers[i];
        auto cmdBufferImpl = getDebugObj(cmdBufferIn);
        auto innerCmdBuffer = getInnerObj(cmdBufferIn);
        innerCommandBuffers.add(innerCmdBuffer);
        if (cmdBufferImpl->isOpen)
        {
            GFX_DIAGNOSE_ERROR_FORMAT(
                "Command buffer %lld is still open. A command buffer must be closed "
                "before submitting to a command queue.",
                cmdBufferImpl->uid);
        }
        if (i > 0)
        {
            if (cmdBufferImpl->m_transientHeap != getDebugObj(commandBuffers[0])->m_transientHeap)
            {
                GFX_DIAGNOSE_ERROR("Command buffers passed to a single executeCommandBuffers "
                                   "call must be allocated from the same transient heap.");
            }
        }
    }
    baseObject->executeCommandBuffers(
        count,
        innerCommandBuffers.getBuffer(),
        getInnerObj(fence),
        valueToSignal);
    if (fence)
    {
        getDebugObj(fence)->maxValueToSignal =
            Math::Max(getDebugObj(fence)->maxValueToSignal, valueToSignal);
    }
}

void DebugCommandQueue::waitOnHost()
{
    SLANG_GFX_API_FUNC;
    baseObject->waitOnHost();
}

Result DebugCommandQueue::waitForFenceValuesOnDevice(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* waitValues)
{
    SLANG_GFX_API_FUNC;
    List<IFence*> innerFences;
    for (GfxIndex i = 0; i < fenceCount; ++i)
    {
        innerFences.add(getInnerObj(fences[i]));
    }
    return baseObject->waitForFenceValuesOnDevice(fenceCount, innerFences.getBuffer(), waitValues);
}

Result DebugCommandQueue::getNativeHandle(InteropHandle* outHandle)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getNativeHandle(outHandle);
}

} // namespace debug
} // namespace gfx
