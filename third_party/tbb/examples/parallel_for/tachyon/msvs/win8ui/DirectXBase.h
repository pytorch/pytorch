/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#pragma once

#include <wrl/client.h>
#include <d3d11_1.h>
#include <d2d1_1.h>
#include <d2d1effects.h>
#include <dwrite_1.h>
#include <wincodec.h>
#include "App.xaml.h"
#include <agile.h>

#pragma warning (disable: 4449)

// Helper utilities to make DirectX APIs work with exceptions
namespace DX
{
    inline void ThrowIfFailed(HRESULT hr)
    {
        if (FAILED(hr))
        {
            // Set a breakpoint on this line to catch DirectX API errors
            throw Platform::Exception::CreateException(hr);
        }
    }
}

// Helper class that initializes DirectX APIs
ref class DirectXBase abstract
{
internal:
    DirectXBase();

public:
    virtual void Initialize(Windows::UI::Core::CoreWindow^ window, Windows::UI::Xaml::Controls::SwapChainBackgroundPanel^ panel, float dpi);
    virtual void CreateDeviceIndependentResources();
    virtual void CreateDeviceResources();
    virtual void SetDpi(float dpi);
    virtual void CreateWindowSizeDependentResources();
    virtual void UpdateForWindowSizeChange();
    virtual void Render() = 0;
    virtual void Present();
    virtual float ConvertDipsToPixels(float dips);

protected private:

    Platform::Agile<Windows::UI::Core::CoreWindow>         m_window;
    Windows::UI::Xaml::Controls::SwapChainBackgroundPanel^ m_panel;

    // Direct2D Objects
    Microsoft::WRL::ComPtr<ID2D1Factory1>                  m_d2dFactory;
    Microsoft::WRL::ComPtr<ID2D1Device>                    m_d2dDevice;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext>             m_d2dContext;
    Microsoft::WRL::ComPtr<ID2D1Bitmap1>                   m_d2dTargetBitmap;

    // DirectWrite & Windows Imaging Component Objects
    Microsoft::WRL::ComPtr<IDWriteFactory1>                m_dwriteFactory;
    Microsoft::WRL::ComPtr<IWICImagingFactory2>            m_wicFactory;

    // Direct3D Objects
    Microsoft::WRL::ComPtr<ID3D11Device1>                  m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext1>           m_d3dContext;
    Microsoft::WRL::ComPtr<IDXGISwapChain1>                m_swapChain;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView>         m_renderTargetView;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilView>         m_depthStencilView;

    D3D_FEATURE_LEVEL                                      m_featureLevel;
    Windows::Foundation::Size                              m_renderTargetSize;
    Windows::Foundation::Rect                              m_windowBounds;
    float                                                  m_dpi;
};

#pragma warning (default: 4449)
