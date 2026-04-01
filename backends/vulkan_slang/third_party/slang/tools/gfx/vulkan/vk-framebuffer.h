// vk-framebuffer.h
#pragma once

#include "vk-base.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

enum
{
    kMaxRenderTargets = 8,
    kMaxTargets = kMaxRenderTargets + 1,
};

class FramebufferLayoutImpl : public FramebufferLayoutBase
{
public:
    VkRenderPass m_renderPass;
    DeviceImpl* m_renderer;
    Array<VkAttachmentDescription, kMaxTargets> m_targetDescs;
    Array<VkAttachmentReference, kMaxRenderTargets> m_colorReferences;
    VkAttachmentReference m_depthReference;
    bool m_hasDepthStencilTarget;
    uint32_t m_renderTargetCount;
    VkSampleCountFlagBits m_sampleCount = VK_SAMPLE_COUNT_1_BIT;

public:
    ~FramebufferLayoutImpl();
    Result init(DeviceImpl* renderer, const IFramebufferLayout::Desc& desc);
};

class FramebufferImpl : public FramebufferBase
{
public:
    VkFramebuffer m_handle;
    ShortList<ComPtr<IResourceView>> renderTargetViews;
    ComPtr<IResourceView> depthStencilView;
    uint32_t m_width;
    uint32_t m_height;
    BreakableReference<DeviceImpl> m_renderer;
    VkClearValue m_clearValues[kMaxTargets];
    RefPtr<FramebufferLayoutImpl> m_layout;

public:
    ~FramebufferImpl();

    Result init(DeviceImpl* renderer, const IFramebuffer::Desc& desc);
};

} // namespace vk
} // namespace gfx
