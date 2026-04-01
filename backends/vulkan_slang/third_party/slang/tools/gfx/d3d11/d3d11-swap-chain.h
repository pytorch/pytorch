// d3d11-swap-chain.h
#pragma once

#include "d3d11-base.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class SwapchainImpl : public D3DSwapchainBase
{
public:
    ComPtr<ID3D11Device> m_device;
    ComPtr<IDXGIFactory> m_dxgiFactory;
    RefPtr<DeviceImpl> m_renderer;
    Result init(DeviceImpl* renderer, const ISwapchain::Desc& swapchainDesc, WindowHandle window);

    virtual void createSwapchainBufferImages() override;
    virtual IDXGIFactory* getDXGIFactory() override { return m_dxgiFactory; }
    virtual IUnknown* getOwningDevice() override { return m_device; }
    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) override;
    virtual SLANG_NO_THROW bool SLANG_MCALL isOccluded() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setFullScreenMode(bool mode) override;
};

} // namespace d3d11
} // namespace gfx
