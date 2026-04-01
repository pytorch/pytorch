// metal-framebuffer.cpp
#include "metal-framebuffer.h"

#include "metal-device.h"
#include "metal-helper-functions.h"
#include "metal-resource-views.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

Result FramebufferLayoutImpl::init(const IFramebufferLayout::Desc& desc)
{
    for (Index i = 0; i < desc.renderTargetCount; ++i)
    {
        m_renderTargets.add(desc.renderTargets[i]);
    }
    if (desc.depthStencil)
    {
        m_depthStencil = *desc.depthStencil;
    }
    else
    {
        m_depthStencil = {};
    }
    return SLANG_OK;
}

Result FramebufferImpl::init(DeviceImpl* device, const IFramebuffer::Desc& desc)
{
    m_device = device;
    m_layout = static_cast<FramebufferLayoutImpl*>(desc.layout);
    m_renderTargetViews.setCount(desc.renderTargetCount);
    for (Index i = 0; i < desc.renderTargetCount; ++i)
    {
        m_renderTargetViews[i] = static_cast<TextureResourceViewImpl*>(desc.renderTargetViews[i]);
    }
    m_depthStencilView = static_cast<TextureResourceViewImpl*>(desc.depthStencilView);

    // Determine framebuffer dimensions & sample count;
    m_width = 1;
    m_height = 1;
    m_sampleCount = 1;

    auto visitView = [this](TextureResourceViewImpl* view)
    {
        const ITextureResource::Desc* textureDesc = view->m_texture->getDesc();
        const IResourceView::Desc* viewDesc = view->getViewDesc();
        m_width =
            Math::Max(1u, uint32_t(textureDesc->size.width >> viewDesc->subresourceRange.mipLevel));
        m_height = Math::Max(
            1u,
            uint32_t(textureDesc->size.height >> viewDesc->subresourceRange.mipLevel));
        m_sampleCount = Math::Max(m_sampleCount, uint32_t(textureDesc->sampleDesc.numSamples));
        return SLANG_OK;
    };

    for (auto view : m_renderTargetViews)
    {
        visitView(view);
    }
    if (m_depthStencilView)
    {
        visitView(m_depthStencilView);
    }

    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
