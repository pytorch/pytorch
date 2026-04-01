// d3d11-swap-chain.cpp
#include "d3d11-swap-chain.h"

#include "d3d11-device.h"
#include "d3d11-texture.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

Result SwapchainImpl::init(
    DeviceImpl* renderer,
    const ISwapchain::Desc& swapchainDesc,
    WindowHandle window)
{
    m_renderer = renderer;
    m_device = renderer->m_device;
    m_dxgiFactory = renderer->m_dxgiFactory;
    return D3DSwapchainBase::init(swapchainDesc, window, DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL);
}

void SwapchainImpl::createSwapchainBufferImages()
{
    m_images.clear();
    // D3D11 implements automatic back buffer rotation, so the application
    // always render to buffer 0.
    ComPtr<ID3D11Resource> d3dResource;
    m_swapChain->GetBuffer(0, IID_PPV_ARGS(d3dResource.writeRef()));
    ITextureResource::Desc imageDesc = {};
    imageDesc.type = IResource::Type::Texture2D;
    imageDesc.arraySize = 0;
    imageDesc.numMipLevels = 1;
    imageDesc.size.width = m_desc.width;
    imageDesc.size.height = m_desc.height;
    imageDesc.size.depth = 1;
    imageDesc.format = m_desc.format;
    imageDesc.defaultState = ResourceState::Present;
    imageDesc.allowedStates = ResourceStateSet(
        ResourceState::Present,
        ResourceState::CopyDestination,
        ResourceState::RenderTarget);
    RefPtr<TextureResourceImpl> image = new TextureResourceImpl(imageDesc);
    image->m_resource = d3dResource;
    for (GfxIndex i = 0; i < m_desc.imageCount; i++)
    {
        m_images.add(image);
    }
}

SLANG_NO_THROW Result SLANG_MCALL SwapchainImpl::resize(GfxCount width, GfxCount height)
{
    m_renderer->m_currentFramebuffer = nullptr;
    m_renderer->m_immediateContext->ClearState();
    return D3DSwapchainBase::resize(width, height);
}

SLANG_NO_THROW bool SLANG_MCALL SwapchainImpl::isOccluded()
{
    return false;
}

SLANG_NO_THROW Result SLANG_MCALL SwapchainImpl::setFullScreenMode(bool mode)
{
    return SLANG_FAIL;
}

} // namespace d3d11
} // namespace gfx
