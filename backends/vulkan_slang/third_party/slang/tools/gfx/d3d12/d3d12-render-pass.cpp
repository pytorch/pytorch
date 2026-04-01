// d3d12-render-pass.cpp
#include "d3d12-render-pass.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

void RenderPassLayoutImpl::init(const IRenderPassLayout::Desc& desc)
{
    SimpleRenderPassLayout::init(desc);
    m_framebufferLayout = static_cast<FramebufferLayoutImpl*>(desc.framebufferLayout);
    m_hasDepthStencil = m_framebufferLayout->m_hasDepthStencil;
}

} // namespace d3d12
} // namespace gfx
