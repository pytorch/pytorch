// vk-command-queue.h
#pragma once

#include "vk-base.h"
#include "vk-device.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class CommandQueueImpl : public ICommandQueue, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandQueue* getInterface(const Guid& guid);

public:
    Desc m_desc;
    RefPtr<DeviceImpl> m_renderer;
    VkQueue m_queue;
    uint32_t m_queueFamilyIndex;
    struct FenceWaitInfo
    {
        RefPtr<FenceImpl> fence;
        uint64_t waitValue;
    };
    List<FenceWaitInfo> m_pendingWaitFences;
    VkSemaphore m_pendingWaitSemaphores[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    List<VkCommandBuffer> m_submitCommandBuffers;
    VkSemaphore m_semaphore;
    ~CommandQueueImpl();

    void init(DeviceImpl* renderer, VkQueue queue, uint32_t queueFamilyIndex);

    virtual SLANG_NO_THROW void SLANG_MCALL waitOnHost() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;

    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    waitForFenceValuesOnDevice(GfxCount fenceCount, IFence** fences, uint64_t* waitValues) override;

    void queueSubmitImpl(
        uint32_t count,
        ICommandBuffer* const* commandBuffers,
        IFence* fence,
        uint64_t valueToSignal);

    virtual SLANG_NO_THROW void SLANG_MCALL executeCommandBuffers(
        GfxCount count,
        ICommandBuffer* const* commandBuffers,
        IFence* fence,
        uint64_t valueToSignal) override;
};

} // namespace vk
} // namespace gfx
