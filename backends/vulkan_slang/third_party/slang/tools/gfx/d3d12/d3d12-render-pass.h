// d3d12-render-pass.h
#pragma once

#include "d3d12-base.h"
#include "d3d12-framebuffer.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class RenderPassLayoutImpl : public SimpleRenderPassLayout
{
public:
    RefPtr<FramebufferLayoutImpl> m_framebufferLayout;
    void init(const IRenderPassLayout::Desc& desc);
};

} // namespace d3d12
} // namespace gfx
