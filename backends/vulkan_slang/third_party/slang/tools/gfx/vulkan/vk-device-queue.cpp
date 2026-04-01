// vk-device-queue.cpp
#include "vk-device-queue.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

namespace gfx
{
using namespace Slang;

VulkanDeviceQueue::~VulkanDeviceQueue()
{
    destroy();
}

void VulkanDeviceQueue::destroy()
{
    if (m_api)
    {
        for (int i = 0; i < int(EventType::CountOf); ++i)
        {
            m_api->vkDestroySemaphore(m_api->m_device, m_semaphores[i], nullptr);
        }

        for (int i = 0; i < m_numCommandBuffers; i++)
        {
            m_api
                ->vkFreeCommandBuffers(m_api->m_device, m_commandPools[i], 1, &m_commandBuffers[i]);
            m_api->vkDestroyFence(m_api->m_device, m_fences[i].fence, nullptr);
            m_api->vkDestroyCommandPool(m_api->m_device, m_commandPools[i], nullptr);
        }
        m_api = nullptr;
    }
}

SlangResult VulkanDeviceQueue::init(const VulkanApi& api, VkQueue queue, int queueIndex)
{
    assert(m_api == nullptr);

    for (int i = 0; i < int(EventType::CountOf); ++i)
    {
        m_semaphores[i] = VK_NULL_HANDLE;
        m_currentSemaphores[i] = VK_NULL_HANDLE;
    }

    m_numCommandBuffers = kMaxCommandBuffers;
    m_queueIndex = queueIndex;

    m_queue = queue;

    for (int i = 0; i < m_numCommandBuffers; i++)
    {
        VkCommandPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        poolCreateInfo.queueFamilyIndex = queueIndex;

        api.vkCreateCommandPool(api.m_device, &poolCreateInfo, nullptr, &m_commandPools[i]);

        VkCommandBufferAllocateInfo commandInfo = {};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandInfo.commandPool = m_commandPools[i];
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;

        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0; // VK_FENCE_CREATE_SIGNALED_BIT;
        Fence& fence = m_fences[i];

        api.vkAllocateCommandBuffers(api.m_device, &commandInfo, &m_commandBuffers[i]);

        api.vkCreateFence(api.m_device, &fenceCreateInfo, nullptr, &fence.fence);
        fence.active = false;
        fence.value = 0;
    }

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (int i = 0; i < int(EventType::CountOf); ++i)
    {
        api.vkCreateSemaphore(api.m_device, &semaphoreCreateInfo, nullptr, &m_semaphores[i]);
    }

    // Set the api - also marks that the queue appears to be valid
    m_api = &api;

    // Second step of flush to prime command buffer
    flushStepB();

    return SLANG_OK;
}

void VulkanDeviceQueue::flushStepA()
{
    m_api->vkEndCommandBuffer(m_commandBuffer);

    VkPipelineStageFlags stageFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Wait semaphores
    if (isCurrent(EventType::BeginFrame))
    {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_currentSemaphores[int(EventType::BeginFrame)];
    }

    submitInfo.pWaitDstStageMask = &stageFlags;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    // Signal semaphores
    if (isCurrent(EventType::EndFrame))
    {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_currentSemaphores[int(EventType::EndFrame)];
    }

    Fence& fence = m_fences[m_commandBufferIndex];

    m_api->vkQueueSubmit(m_queue, 1, &submitInfo, fence.fence);

    // mark signaled fence value
    fence.value = m_nextFenceValue;
    fence.active = true;

    // increment fence value
    m_nextFenceValue++;

    // No longer waiting on this semaphore
    makeCompleted(EventType::BeginFrame);
    makeCompleted(EventType::EndFrame);
}

void VulkanDeviceQueue::_updateFenceAtIndex(int fenceIndex, bool blocking)
{
    Fence& fence = m_fences[fenceIndex];

    if (fence.active)
    {
        uint64_t timeout = blocking ? ~uint64_t(0) : 0;

        if (VK_SUCCESS ==
            m_api->vkWaitForFences(m_api->m_device, 1, &fence.fence, VK_TRUE, timeout))
        {
            m_api->vkResetFences(m_api->m_device, 1, &fence.fence);

            fence.active = false;

            if (fence.value > m_lastFenceCompleted)
            {
                m_lastFenceCompleted = fence.value;
            }
        }
    }
}

void VulkanDeviceQueue::flushStepB()
{
    m_commandBufferIndex = (m_commandBufferIndex + 1) % m_numCommandBuffers;
    m_commandBuffer = m_commandBuffers[m_commandBufferIndex];
    m_commandPool = m_commandPools[m_commandBufferIndex];

    // non-blocking update of fence values
    for (int i = 0; i < m_numCommandBuffers; ++i)
    {
        _updateFenceAtIndex(i, false);
    }

    // blocking update of fence values
    _updateFenceAtIndex(m_commandBufferIndex, true);

    m_api->vkResetCommandPool(m_api->m_device, m_commandPool, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    m_api->vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
}

void VulkanDeviceQueue::flush()
{
    flushStepA();
    flushStepB();
}

void VulkanDeviceQueue::flushAndWait()
{
    flush();
    waitForIdle();
}

VkSemaphore VulkanDeviceQueue::getSemaphore(EventType eventType)
{
    return m_semaphores[int(eventType)];
}

VkSemaphore VulkanDeviceQueue::makeCurrent(EventType eventType)
{
    assert(!isCurrent(eventType));
    VkSemaphore semaphore = m_semaphores[int(eventType)];
    m_currentSemaphores[int(eventType)] = semaphore;
    return semaphore;
}

void VulkanDeviceQueue::makeCompleted(EventType eventType)
{
    m_currentSemaphores[int(eventType)] = VK_NULL_HANDLE;
}

} // namespace gfx
