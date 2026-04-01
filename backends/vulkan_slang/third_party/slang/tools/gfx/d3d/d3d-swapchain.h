#pragma once

#include "../renderer-shared.h"
#include "core/slang-basic.h"
#include "d3d-util.h"
#include "slang-gfx.h"

#include <dxgi1_4.h>

namespace gfx
{
class D3DSwapchainBase : public ISwapchain, public Slang::ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ISwapchain* getInterface(const Slang::Guid& guid)
    {
        if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ISwapchain)
            return static_cast<ISwapchain*>(this);
        return nullptr;
    }

public:
    Result init(const ISwapchain::Desc& desc, WindowHandle window, DXGI_SWAP_EFFECT swapEffect)
    {
        // Return fail on non-supported platforms.
        switch (window.type)
        {
        case WindowHandle::Type::Win32Handle:
            break;
        default:
            return SLANG_FAIL;
        }

        m_desc = desc;

        m_desc.format = srgbToLinearFormat(m_desc.format);

        // Describe the swap chain.
        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        swapChainDesc.BufferCount = desc.imageCount;
        swapChainDesc.BufferDesc.Width = desc.width;
        swapChainDesc.BufferDesc.Height = desc.height;
        swapChainDesc.BufferDesc.Format = D3DUtil::getMapFormat(m_desc.format);
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = swapEffect;
        swapChainDesc.OutputWindow = (HWND)window.handleValues[0];
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.Windowed = TRUE;
        if (!desc.enableVSync)
        {
            swapChainDesc.Flags |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        }

        // Swap chain needs the queue so that it can force a flush on it.
        ComPtr<IDXGIFactory2> dxgiFactory2;
        getDXGIFactory()->QueryInterface(IID_PPV_ARGS(dxgiFactory2.writeRef()));
        if (!dxgiFactory2)
        {
            ComPtr<IDXGISwapChain> swapChain;
            SLANG_RETURN_ON_FAIL(getDXGIFactory()->CreateSwapChain(
                getOwningDevice(),
                &swapChainDesc,
                swapChain.writeRef()));
            SLANG_RETURN_ON_FAIL(getDXGIFactory()->MakeWindowAssociation(
                (HWND)window.handleValues[0],
                DXGI_MWA_NO_ALT_ENTER));
            SLANG_RETURN_ON_FAIL(swapChain->QueryInterface(m_swapChain.writeRef()));
        }
        else
        {
            DXGI_SWAP_CHAIN_DESC1 desc1 = {};
            desc1.BufferCount = swapChainDesc.BufferCount;
            desc1.BufferUsage = swapChainDesc.BufferUsage;
            desc1.Flags = swapChainDesc.Flags;
            desc1.Format = swapChainDesc.BufferDesc.Format;
            desc1.Height = swapChainDesc.BufferDesc.Height;
            desc1.Width = swapChainDesc.BufferDesc.Width;
            desc1.SampleDesc = swapChainDesc.SampleDesc;
            desc1.SwapEffect = swapChainDesc.SwapEffect;
            ComPtr<IDXGISwapChain1> swapChain1;
            SLANG_RETURN_ON_FAIL(dxgiFactory2->CreateSwapChainForHwnd(
                getOwningDevice(),
                (HWND)window.handleValues[0],
                &desc1,
                nullptr,
                nullptr,
                swapChain1.writeRef()));
            SLANG_RETURN_ON_FAIL(swapChain1->QueryInterface(m_swapChain.writeRef()));
        }

        createSwapchainBufferImages();
        return SLANG_OK;
    }
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override { return m_desc; }
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getImage(GfxIndex index, ITextureResource** outResource) override
    {
        returnComPtr(outResource, m_images[index]);
        return SLANG_OK;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL present() override
    {
        const auto res = m_swapChain->Present(m_desc.enableVSync ? 1 : 0, 0);

        // We may want to wait for crash dump completion for some kinds of debugging scenarios
        if (res == DXGI_ERROR_DEVICE_REMOVED || res == DXGI_ERROR_DEVICE_RESET)
        {
            D3DUtil::waitForCrashDumpCompletion(res);
        }

        if (SLANG_FAILED(res))
        {
            return SLANG_FAIL;
        }
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW int SLANG_MCALL acquireNextImage() override
    {
        uint32_t count;
        m_swapChain->GetLastPresentCount(&count);
        return (int)(count % m_desc.imageCount);
    }


    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) override
    {
        if (width == m_desc.width && height == m_desc.height)
            return SLANG_OK;

        m_desc.width = width;
        m_desc.height = height;
        for (auto& image : m_images)
            image = nullptr;
        m_images.clear();
        auto result = m_swapChain->ResizeBuffers(
            m_desc.imageCount,
            width,
            height,
            D3DUtil::getMapFormat(m_desc.format),
            m_desc.enableVSync ? 0 : DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        if (result != 0)
            return SLANG_FAIL;
        createSwapchainBufferImages();
        return SLANG_OK;
    }

public:
    virtual void createSwapchainBufferImages() = 0;
    virtual IDXGIFactory* getDXGIFactory() = 0;
    virtual IUnknown* getOwningDevice() = 0;
    ISwapchain::Desc m_desc;
    ComPtr<IDXGISwapChain2> m_swapChain;
    Slang::ShortList<Slang::RefPtr<TextureResource>> m_images;
};

} // namespace gfx
