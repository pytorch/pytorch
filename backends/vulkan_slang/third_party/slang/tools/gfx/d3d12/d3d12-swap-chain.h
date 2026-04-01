// d3d12-swap-chain.h
#pragma once

#include "d3d12-base.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

static const Int kMaxNumRenderFrames = 4;

class SwapchainImpl : public D3DSwapchainBase
{
public:
    ComPtr<ID3D12CommandQueue> m_queue;
    ComPtr<IDXGIFactory> m_dxgiFactory;
    ComPtr<IDXGISwapChain3> m_swapChain3;
    ComPtr<ID3D12Fence> m_fence;
    ShortList<HANDLE, kMaxNumRenderFrames> m_frameEvents;
    uint64_t fenceValue = 0;
    Result init(DeviceImpl* renderer, const ISwapchain::Desc& swapchainDesc, WindowHandle window);

    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) override;

    virtual void createSwapchainBufferImages() override;
    virtual IDXGIFactory* getDXGIFactory() override { return m_dxgiFactory; }
    virtual IUnknown* getOwningDevice() override { return m_queue; }
    virtual SLANG_NO_THROW int SLANG_MCALL acquireNextImage() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL present() override;
    virtual SLANG_NO_THROW bool SLANG_MCALL isOccluded() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setFullScreenMode(bool mode) override;
};

} // namespace d3d12
} // namespace gfx
