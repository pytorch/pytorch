#include "simple-render-pass-layout.h"

#include "renderer-shared.h"

namespace gfx
{

IRenderPassLayout* SimpleRenderPassLayout::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IRenderPassLayout)
        return static_cast<IRenderPassLayout*>(this);
    return nullptr;
}

void SimpleRenderPassLayout::init(const IRenderPassLayout::Desc& desc)
{
    m_renderTargetAccesses.setCount(desc.renderTargetCount);
    for (GfxIndex i = 0; i < desc.renderTargetCount; i++)
        m_renderTargetAccesses[i] = desc.renderTargetAccess[i];
    m_hasDepthStencil = (desc.depthStencilAccess != nullptr);
    if (m_hasDepthStencil)
        m_depthStencilAccess = *desc.depthStencilAccess;
}

} // namespace gfx
