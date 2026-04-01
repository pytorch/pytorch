// vk-render-pass.cpp
#include "vk-render-pass.h"

#include "vk-helper-functions.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

IRenderPassLayout* RenderPassLayoutImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IRenderPassLayout)
        return static_cast<IRenderPassLayout*>(this);
    return nullptr;
}

RenderPassLayoutImpl::~RenderPassLayoutImpl()
{
    m_renderer->m_api.vkDestroyRenderPass(m_renderer->m_api.m_device, m_renderPass, nullptr);
}

Result RenderPassLayoutImpl::init(DeviceImpl* renderer, const IRenderPassLayout::Desc& desc)
{
    m_renderer = renderer;

    // Create render pass using load/storeOp and layouts info from `desc`.
    auto framebufferLayout = static_cast<FramebufferLayoutImpl*>(desc.framebufferLayout);
    assert(desc.renderTargetCount == framebufferLayout->m_renderTargetCount);

    // We need extra space if we have depth buffer
    Array<VkAttachmentDescription, kMaxTargets> targetDescs;
    targetDescs = framebufferLayout->m_targetDescs;
    for (GfxIndex i = 0; i < desc.renderTargetCount; ++i)
    {
        VkAttachmentDescription& dst = targetDescs[i];
        auto access = desc.renderTargetAccess[i];
        // Fill in loadOp/storeOp and layout from desc.
        dst.loadOp = translateLoadOp(access.loadOp);
        dst.storeOp = translateStoreOp(access.storeOp);
        dst.stencilLoadOp = translateLoadOp(access.stencilLoadOp);
        dst.stencilStoreOp = translateStoreOp(access.stencilStoreOp);
        dst.initialLayout = VulkanUtil::mapResourceStateToLayout(access.initialState);
        dst.finalLayout = VulkanUtil::mapResourceStateToLayout(access.finalState);
    }

    if (framebufferLayout->m_hasDepthStencilTarget)
    {
        VkAttachmentDescription& dst = targetDescs[desc.renderTargetCount];
        auto access = *desc.depthStencilAccess;
        dst.loadOp = translateLoadOp(access.loadOp);
        dst.storeOp = translateStoreOp(access.storeOp);
        dst.stencilLoadOp = translateLoadOp(access.stencilLoadOp);
        dst.stencilStoreOp = translateStoreOp(access.stencilStoreOp);
        dst.initialLayout = VulkanUtil::mapResourceStateToLayout(access.initialState);
        dst.finalLayout = VulkanUtil::mapResourceStateToLayout(access.finalState);
    }

    VkSubpassDescription subpassDesc = {};
    subpassDesc.flags = 0;
    subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDesc.inputAttachmentCount = 0u;
    subpassDesc.pInputAttachments = nullptr;
    subpassDesc.colorAttachmentCount = desc.renderTargetCount;
    subpassDesc.pColorAttachments = framebufferLayout->m_colorReferences.getBuffer();
    subpassDesc.pResolveAttachments = nullptr;
    subpassDesc.pDepthStencilAttachment =
        framebufferLayout->m_hasDepthStencilTarget ? &framebufferLayout->m_depthReference : nullptr;
    subpassDesc.preserveAttachmentCount = 0u;
    subpassDesc.pPreserveAttachments = nullptr;

    VkRenderPassCreateInfo renderPassCreateInfo = {};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = (uint32_t)targetDescs.getCount();
    renderPassCreateInfo.pAttachments = targetDescs.getBuffer();
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDesc;
    SLANG_VK_RETURN_ON_FAIL(m_renderer->m_api.vkCreateRenderPass(
        m_renderer->m_api.m_device,
        &renderPassCreateInfo,
        nullptr,
        &m_renderPass));
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
