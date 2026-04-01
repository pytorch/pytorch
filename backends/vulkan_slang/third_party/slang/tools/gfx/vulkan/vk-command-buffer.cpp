// vk-command-buffer.cpp
#include "vk-command-buffer.h"

#include "vk-device.h"
#include "vk-shader-object.h"
#include "vk-util.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

// There are a pair of cyclic references between a `TransientResourceHeap` and
// a `CommandBuffer` created from the heap. We need to break the cycle when
// the public reference count of a command buffer drops to 0.

ICommandBuffer* CommandBufferImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandBuffer)
        return static_cast<ICommandBuffer*>(this);
    return nullptr;
}

void CommandBufferImpl::comFree()
{
    m_transientHeap.breakStrongReference();
}

Result CommandBufferImpl::init(
    DeviceImpl* renderer,
    VkCommandPool pool,
    TransientResourceHeapImpl* transientHeap)
{
    m_renderer = renderer;
    m_transientHeap = transientHeap;
    m_pool = pool;

    auto& api = renderer->m_api;
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    SLANG_VK_RETURN_ON_FAIL(
        api.vkAllocateCommandBuffers(api.m_device, &allocInfo, &m_commandBuffer));

    beginCommandBuffer();
    return SLANG_OK;
}

void CommandBufferImpl::beginCommandBuffer()
{
    auto& api = m_renderer->m_api;
    VkCommandBufferBeginInfo beginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    api.vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    if (m_preCommandBuffer)
    {
        api.vkBeginCommandBuffer(m_preCommandBuffer, &beginInfo);
    }
    m_isPreCommandBufferEmpty = true;
}

Result CommandBufferImpl::createPreCommandBuffer()
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    auto& api = m_renderer->m_api;
    SLANG_VK_RETURN_ON_FAIL(
        api.vkAllocateCommandBuffers(api.m_device, &allocInfo, &m_preCommandBuffer));
    VkCommandBufferBeginInfo beginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    api.vkBeginCommandBuffer(m_preCommandBuffer, &beginInfo);
    return SLANG_OK;
}

VkCommandBuffer CommandBufferImpl::getPreCommandBuffer()
{
    m_isPreCommandBufferEmpty = false;
    if (m_preCommandBuffer)
        return m_preCommandBuffer;
    createPreCommandBuffer();
    return m_preCommandBuffer;
}

void CommandBufferImpl::encodeRenderCommands(
    IRenderPassLayout* renderPass,
    IFramebuffer* framebuffer,
    IRenderCommandEncoder** outEncoder)
{
    if (!m_renderCommandEncoder)
    {
        m_renderCommandEncoder = new RenderCommandEncoder();
        m_renderCommandEncoder->init(this);
    }
    m_renderCommandEncoder->beginPass(renderPass, framebuffer);
    *outEncoder = m_renderCommandEncoder.Ptr();
}

void CommandBufferImpl::encodeComputeCommands(IComputeCommandEncoder** outEncoder)
{
    if (!m_computeCommandEncoder)
    {
        m_computeCommandEncoder = new ComputeCommandEncoder();
        m_computeCommandEncoder->init(this);
    }
    *outEncoder = m_computeCommandEncoder.Ptr();
}

void CommandBufferImpl::encodeResourceCommands(IResourceCommandEncoder** outEncoder)
{
    if (!m_resourceCommandEncoder)
    {
        m_resourceCommandEncoder = new ResourceCommandEncoder();
        m_resourceCommandEncoder->init(this);
    }
    *outEncoder = m_resourceCommandEncoder.Ptr();
}

void CommandBufferImpl::encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder)
{
    if (!m_rayTracingCommandEncoder)
    {
        if (m_renderer->m_api.vkCmdBuildAccelerationStructuresKHR)
        {
            m_rayTracingCommandEncoder = new RayTracingCommandEncoder();
            m_rayTracingCommandEncoder->init(this);
        }
    }
    *outEncoder = m_rayTracingCommandEncoder.Ptr();
}

void CommandBufferImpl::close()
{
    auto& vkAPI = m_renderer->m_api;
    if (!m_isPreCommandBufferEmpty)
    {
        // `preCmdBuffer` contains buffer transfer commands for shader object
        // uniform buffers, and we need a memory barrier here to ensure the
        // transfers are visible to shaders.
        VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkAPI.vkCmdPipelineBarrier(
            m_preCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0,
            1,
            &memBarrier,
            0,
            nullptr,
            0,
            nullptr);
        vkAPI.vkEndCommandBuffer(m_preCommandBuffer);
    }
    vkAPI.vkEndCommandBuffer(m_commandBuffer);
}

Result CommandBufferImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Vulkan;
    outHandle->handleValue = (uint64_t)m_commandBuffer;
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
