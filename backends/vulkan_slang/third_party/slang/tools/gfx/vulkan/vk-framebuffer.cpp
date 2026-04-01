// vk-framebuffer.cpp
#include "vk-framebuffer.h"

#include "vk-device.h"
#include "vk-helper-functions.h"
#include "vk-resource-views.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

FramebufferLayoutImpl::~FramebufferLayoutImpl()
{
    m_renderer->m_api.vkDestroyRenderPass(m_renderer->m_api.m_device, m_renderPass, nullptr);
}

Result FramebufferLayoutImpl::init(DeviceImpl* renderer, const IFramebufferLayout::Desc& desc)
{
    m_renderer = renderer;
    m_renderTargetCount = desc.renderTargetCount;
    // Create render pass.
    int numTargets = m_renderTargetCount;
    m_hasDepthStencilTarget = (desc.depthStencil != nullptr);
    if (m_hasDepthStencilTarget)
    {
        numTargets++;
    }
    // We need extra space if we have depth buffer
    m_targetDescs.setCount(numTargets);
    for (GfxIndex i = 0; i < desc.renderTargetCount; ++i)
    {
        auto& renderTarget = desc.renderTargets[i];
        VkAttachmentDescription& dst = m_targetDescs[i];

        dst.flags = 0;
        dst.format = VulkanUtil::getVkFormat(renderTarget.format);
        if (renderTarget.format == Format::Unknown)
            dst.format = VK_FORMAT_R8G8B8A8_UNORM;
        dst.samples = (VkSampleCountFlagBits)renderTarget.sampleCount;

        // The following load/store/layout settings does not matter.
        // In FramebufferLayout we just need a "compatible" render pass that
        // can be used to create a framebuffer. A framebuffer created
        // with this render pass setting can be used with actual render passes
        // that has a different loadOp/storeOp/layout setting.
        dst.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        dst.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        dst.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        dst.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        dst.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        dst.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        m_sampleCount = Math::Max(dst.samples, m_sampleCount);
    }

    if (desc.depthStencil)
    {
        VkAttachmentDescription& dst = m_targetDescs[desc.renderTargetCount];
        dst.flags = 0;
        dst.format = VulkanUtil::getVkFormat(desc.depthStencil->format);
        dst.samples = (VkSampleCountFlagBits)desc.depthStencil->sampleCount;
        dst.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        dst.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        dst.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        dst.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        dst.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        dst.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        m_sampleCount = Math::Max(dst.samples, m_sampleCount);
    }

    Array<VkAttachmentReference, kMaxRenderTargets>& colorReferences = m_colorReferences;
    colorReferences.setCount(desc.renderTargetCount);
    for (GfxIndex i = 0; i < desc.renderTargetCount; ++i)
    {
        VkAttachmentReference& dst = colorReferences[i];
        dst.attachment = i;
        dst.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    m_depthReference = VkAttachmentReference{};
    m_depthReference.attachment = desc.renderTargetCount;
    m_depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDesc = {};
    subpassDesc.flags = 0;
    subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDesc.inputAttachmentCount = 0u;
    subpassDesc.pInputAttachments = nullptr;
    subpassDesc.colorAttachmentCount = desc.renderTargetCount;
    subpassDesc.pColorAttachments = colorReferences.getBuffer();
    subpassDesc.pResolveAttachments = nullptr;
    subpassDesc.pDepthStencilAttachment = m_hasDepthStencilTarget ? &m_depthReference : nullptr;
    subpassDesc.preserveAttachmentCount = 0u;
    subpassDesc.pPreserveAttachments = nullptr;

    VkRenderPassCreateInfo renderPassCreateInfo = {};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = numTargets;
    renderPassCreateInfo.pAttachments = m_targetDescs.getBuffer();
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDesc;
    SLANG_VK_RETURN_ON_FAIL(m_renderer->m_api.vkCreateRenderPass(
        m_renderer->m_api.m_device,
        &renderPassCreateInfo,
        nullptr,
        &m_renderPass));
    return SLANG_OK;
}

FramebufferImpl::~FramebufferImpl()
{
    m_renderer->m_api.vkDestroyFramebuffer(m_renderer->m_api.m_device, m_handle, nullptr);
}

Result FramebufferImpl::init(DeviceImpl* renderer, const IFramebuffer::Desc& desc)
{
    m_renderer = renderer;
    uint32_t layerCount = 0;

    auto dsv = desc.depthStencilView ? static_cast<TextureResourceViewImpl*>(desc.depthStencilView)
                                     : nullptr;
    // Get frame dimensions from attachments.
    if (dsv)
    {
        // If we have a depth attachment, get frame size from there.
        auto size = dsv->m_texture->getDesc()->size;
        auto viewDesc = dsv->getViewDesc();
        m_width = getMipLevelSize(viewDesc->subresourceRange.mipLevel, size.width);
        m_height = getMipLevelSize(viewDesc->subresourceRange.mipLevel, size.height);
        layerCount = viewDesc->subresourceRange.layerCount;
    }
    else if (desc.renderTargetCount)
    {
        // If we don't have a depth attachment, then we must have at least
        // one color attachment. Get frame dimension from there.
        auto viewImpl = static_cast<TextureResourceViewImpl*>(desc.renderTargetViews[0]);
        auto resourceDesc = viewImpl->m_texture->getDesc();
        auto viewDesc = viewImpl->getViewDesc();
        auto size = resourceDesc->size;
        m_width = getMipLevelSize(viewDesc->subresourceRange.mipLevel, size.width);
        m_height = getMipLevelSize(viewDesc->subresourceRange.mipLevel, size.height);
        layerCount = (resourceDesc->type == IResource::Type::Texture3D)
                         ? size.depth
                         : viewDesc->subresourceRange.layerCount;
    }
    else
    {
        // In case we create an "empty" framebuffer, use the maximum viewport dimensions.
        // This to allow arbitrary viewport sizes when rendering to the empty framebuffer.
        m_width = m_renderer->m_api.m_deviceProperties.limits.maxViewportDimensions[0];
        m_height = m_renderer->m_api.m_deviceProperties.limits.maxViewportDimensions[1];
        layerCount = 1;
    }
    if (layerCount == 0)
        layerCount = 1;
    // Create render pass.
    int numTargets = desc.renderTargetCount;
    if (desc.depthStencilView)
        numTargets++;
    Array<VkImageView, kMaxTargets> imageViews;
    imageViews.setCount(numTargets);
    renderTargetViews.setCount(desc.renderTargetCount);
    for (GfxIndex i = 0; i < desc.renderTargetCount; ++i)
    {
        auto resourceView = static_cast<TextureResourceViewImpl*>(desc.renderTargetViews[i]);
        renderTargetViews[i] = resourceView;
        imageViews[i] = resourceView->m_view;
        if (resourceView->m_texture->getDesc()->optimalClearValue)
        {
            memcpy(
                &m_clearValues[i],
                &resourceView->m_texture->getDesc()->optimalClearValue->color,
                sizeof(gfx::ColorClearValue));
        }
    }

    if (dsv)
    {
        imageViews[desc.renderTargetCount] = dsv->m_view;
        depthStencilView = dsv;
        if (dsv->m_texture->getDesc()->optimalClearValue)
        {
            memcpy(
                &m_clearValues[desc.renderTargetCount],
                &dsv->m_texture->getDesc()->optimalClearValue->depthStencil,
                sizeof(gfx::DepthStencilClearValue));
        }
    }

    // Create framebuffer.
    m_layout = static_cast<FramebufferLayoutImpl*>(desc.layout);
    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_layout->m_renderPass;
    framebufferInfo.attachmentCount = numTargets;
    framebufferInfo.pAttachments = imageViews.getBuffer();
    framebufferInfo.width = m_width;
    framebufferInfo.height = m_height;
    framebufferInfo.layers = layerCount;

    SLANG_VK_RETURN_ON_FAIL(m_renderer->m_api.vkCreateFramebuffer(
        m_renderer->m_api.m_device,
        &framebufferInfo,
        nullptr,
        &m_handle));
    return SLANG_OK;
}

} // namespace vk
} // namespace gfx
