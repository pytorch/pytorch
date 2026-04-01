// vk-transient-heap.cpp
#include "vk-transient-heap.h"

#include "vk-device.h"
#include "vk-util.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

void TransientResourceHeapImpl::advanceFence()
{
    m_fenceIndex++;
    if (m_fenceIndex >= m_fences.getCount())
    {
        m_fences.setCount(m_fenceIndex + 1);
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        m_device->m_api.vkCreateFence(
            m_device->m_api.m_device,
            &fenceCreateInfo,
            nullptr,
            &m_fences[m_fenceIndex]);
    }
}

Result TransientResourceHeapImpl::init(const ITransientResourceHeap::Desc& desc, DeviceImpl* device)
{
    Super::init(
        desc,
        (uint32_t)device->m_api.m_deviceProperties.limits.minUniformBufferOffsetAlignment,
        device);

    m_descSetAllocator.m_api = &device->m_api;

    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex =
        device->getQueueFamilyIndex(ICommandQueue::QueueType::Graphics);
    device->m_api
        .vkCreateCommandPool(device->m_api.m_device, &poolCreateInfo, nullptr, &m_commandPool);

    advanceFence();
    return SLANG_OK;
}

TransientResourceHeapImpl::~TransientResourceHeapImpl()
{
    m_commandBufferPool = decltype(m_commandBufferPool)();
    m_device->m_api.vkDestroyCommandPool(m_device->m_api.m_device, m_commandPool, nullptr);
    for (auto fence : m_fences)
    {
        m_device->m_api.vkDestroyFence(m_device->m_api.m_device, fence, nullptr);
    }
    m_descSetAllocator.close();
}

Result TransientResourceHeapImpl::createCommandBuffer(ICommandBuffer** outCmdBuffer)
{
    if (m_commandBufferAllocId < (uint32_t)m_commandBufferPool.getCount())
    {
        auto result = m_commandBufferPool[m_commandBufferAllocId];
        result->m_transientHeap.establishStrongReference();
        result->beginCommandBuffer();
        m_commandBufferAllocId++;
        returnComPtr(outCmdBuffer, result);
        return SLANG_OK;
    }

    RefPtr<CommandBufferImpl> commandBuffer = new CommandBufferImpl();
    SLANG_RETURN_ON_FAIL(commandBuffer->init(m_device, m_commandPool, this));
    m_commandBufferPool.add(commandBuffer);
    m_commandBufferAllocId++;
    returnComPtr(outCmdBuffer, commandBuffer);
    return SLANG_OK;
}

Result TransientResourceHeapImpl::synchronizeAndReset()
{
    m_commandBufferAllocId = 0;
    auto& api = m_device->m_api;
    if (api.vkWaitForFences(
            api.m_device,
            (uint32_t)m_fences.getCount(),
            m_fences.getBuffer(),
            1,
            UINT64_MAX) != VK_SUCCESS)
    {
        return SLANG_FAIL;
    }
    api.vkResetCommandPool(api.m_device, m_commandPool, 0);
    m_descSetAllocator.reset();
    m_fenceIndex = 0;
    Super::reset();
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
