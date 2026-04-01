// metal-framebuffer.h
#pragma once

#include "metal-base.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

enum
{
    kMaxRenderTargets = 8,
    kMaxTargets = kMaxRenderTargets + 1,
};

class FramebufferLayoutImpl : public FramebufferLayoutBase
{
public:
    List<IFramebufferLayout::TargetLayout> m_renderTargets;
    IFramebufferLayout::TargetLayout m_depthStencil;

public:
    Result init(const IFramebufferLayout::Desc& desc);
};

class FramebufferImpl : public FramebufferBase
{
public:
    BreakableReference<DeviceImpl> m_device;
    RefPtr<FramebufferLayoutImpl> m_layout;
    ShortList<RefPtr<TextureResourceViewImpl>> m_renderTargetViews;
    RefPtr<TextureResourceViewImpl> m_depthStencilView;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_sampleCount;

public:
    Result init(DeviceImpl* device, const IFramebuffer::Desc& desc);
};

} // namespace metal
} // namespace gfx
