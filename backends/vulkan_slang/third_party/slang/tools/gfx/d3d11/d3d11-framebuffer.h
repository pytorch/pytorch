// d3d11-framebuffer.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

enum
{
    kMaxUAVs = 64,
    kMaxRTVs = 8,
};

class FramebufferLayoutImpl : public FramebufferLayoutBase
{
public:
    ShortList<IFramebufferLayout::TargetLayout> m_renderTargets;
    bool m_hasDepthStencil = false;
    IFramebufferLayout::TargetLayout m_depthStencil;
};

class FramebufferImpl : public FramebufferBase
{
public:
    ShortList<RefPtr<RenderTargetViewImpl>, kMaxRTVs> renderTargetViews;
    ShortList<ID3D11RenderTargetView*, kMaxRTVs> d3dRenderTargetViews;
    RefPtr<DepthStencilViewImpl> depthStencilView;
    ID3D11DepthStencilView* d3dDepthStencilView;
};

} // namespace d3d11
} // namespace gfx
