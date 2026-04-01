// metal-render-pass.cpp
#include "metal-render-pass.h"

// #include "metal-helper-functions.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

IRenderPassLayout* RenderPassLayoutImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IRenderPassLayout)
        return static_cast<IRenderPassLayout*>(this);
    return nullptr;
}

static inline MTL::LoadAction translateLoadOp(IRenderPassLayout::TargetLoadOp loadOp)
{
    switch (loadOp)
    {
    case IRenderPassLayout::TargetLoadOp::Load:
        return MTL::LoadActionLoad;
    case IRenderPassLayout::TargetLoadOp::Clear:
        return MTL::LoadActionClear;
    case IRenderPassLayout::TargetLoadOp::DontCare:
        return MTL::LoadActionDontCare;
    default:
        return MTL::LoadAction(0);
    }
}

static inline MTL::StoreAction translateStoreOp(IRenderPassLayout::TargetStoreOp storeOp)
{
    switch (storeOp)
    {
    case IRenderPassLayout::TargetStoreOp::Store:
        return MTL::StoreActionStore;
    case IRenderPassLayout::TargetStoreOp::DontCare:
        return MTL::StoreActionDontCare;
    default:
        return MTL::StoreAction(0);
    }
}

Result RenderPassLayoutImpl::init(DeviceImpl* device, const IRenderPassLayout::Desc& desc)
{
    m_device = device;

    FramebufferLayoutImpl* framebufferLayout =
        static_cast<FramebufferLayoutImpl*>(desc.framebufferLayout);
    assert(framebufferLayout);

    // Initialize render pass descriptor, filling in attachment metadata, but leaving texture data
    // unbound.
    m_renderPassDesc = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());

    m_renderPassDesc->setRenderTargetArrayLength(desc.renderTargetCount);
    for (GfxIndex i = 0; i < desc.renderTargetCount; ++i)
    {
        MTL::RenderPassColorAttachmentDescriptor* colorAttachment =
            m_renderPassDesc->colorAttachments()->object(i);
        colorAttachment->setLoadAction(translateLoadOp(desc.renderTargetAccess[i].loadOp));
        colorAttachment->setStoreAction(translateStoreOp(desc.renderTargetAccess[i].storeOp));
    }

    m_renderPassDesc->depthAttachment()->setLoadAction(
        translateLoadOp(desc.depthStencilAccess->loadOp));
    m_renderPassDesc->depthAttachment()->setStoreAction(
        translateStoreOp(desc.depthStencilAccess->storeOp));

    m_renderPassDesc->stencilAttachment()->setLoadAction(
        translateLoadOp(desc.depthStencilAccess->loadOp));
    m_renderPassDesc->stencilAttachment()->setStoreAction(
        translateStoreOp(desc.depthStencilAccess->storeOp));

    return SLANG_OK;
}

} // namespace metal
} // namespace gfx
