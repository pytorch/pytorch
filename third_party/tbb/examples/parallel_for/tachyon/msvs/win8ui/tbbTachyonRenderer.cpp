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

#include "pch.h"
#include "tbbTachyonRenderer.h"
#include <DirectXMath.h>
#include <process.h>
#include <thread>
#include "../../src/tachyon_video.h"
#include "tbb/tbb.h"

using namespace Microsoft::WRL;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Core;
using namespace DirectX;

tbbTachyonRenderer::tbbTachyonRenderer() :
    m_renderNeeded(true)
{
}

tbbTachyonRenderer::~tbbTachyonRenderer()
{
}

void tbbTachyonRenderer::CreateDeviceIndependentResources()
{
    DirectXBase::CreateDeviceIndependentResources();

    DX::ThrowIfFailed(
        m_dwriteFactory->CreateTextFormat(
        L"Segoe UI",
        nullptr,
        DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        32.0f,
        L"en-US",
        &m_textFormat
        )
        );

    DX::ThrowIfFailed(
        m_textFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING)
        );

}

unsigned int __stdcall example_main(void*);

float g_ratiox, g_ratioy;
extern unsigned int *g_pImg;
extern int g_sizex, g_sizey;
extern int global_xsize, global_ysize;
extern int volatile global_number_of_threads;
extern volatile long global_startTime;
extern volatile long global_elapsedTime;

#define SHOW_TEXT 1

void tbbTachyonRenderer::CreateDeviceResources()
{

    DirectXBase::CreateDeviceResources();

    DX::ThrowIfFailed(
        m_d2dContext->CreateSolidColorBrush(
        D2D1::ColorF(D2D1::ColorF::Green),
        &m_Brush
        )
        );

    D2D1_BITMAP_PROPERTIES1 properties = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET,
        D2D1::PixelFormat(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D2D1_ALPHA_MODE_IGNORE
        )
        );


    //Setting manual rendering size
    global_xsize = 800;
    global_ysize = int(global_xsize/m_window->Bounds.Width*m_window->Bounds.Height);
    D2D1_SIZE_U opacityBitmapSize = D2D1::SizeU(global_xsize, global_ysize);

    DX::ThrowIfFailed(
        m_d2dContext->CreateBitmap(
        opacityBitmapSize,
        (BYTE*)g_pImg,
        sizeof(unsigned int)*g_sizex,
        &properties,
        &m_opacityBitmap
        )
        );

    m_d2dContext->SetTarget(m_opacityBitmap.Get());
    m_d2dContext->BeginDraw();

    m_d2dContext->Clear(D2D1::ColorF(D2D1::ColorF::Black, 0.0f));

    DX::ThrowIfFailed(
        m_d2dContext->EndDraw()
        );

    std::thread* thread_tmp=new std::thread(example_main, (void*)NULL);

}

void tbbTachyonRenderer::CreateWindowSizeDependentResources()
{
    DirectXBase::CreateWindowSizeDependentResources();
}

void tbbTachyonRenderer::Render()
{
    D2D1_SIZE_F size = m_d2dContext->GetSize();

#if SHOW_TEXT
    if (video && video->running)
        global_elapsedTime=(long)(time(NULL)-global_startTime);
    
    Platform::String^ text= "Running in " +
        (global_number_of_threads == tbb::task_scheduler_init::automatic? "all hardware threads: ":
            global_number_of_threads.ToString() + (global_number_of_threads==1?" thread: ":" threads: ")) +
        global_elapsedTime.ToString() + (global_elapsedTime>1?" seconds":" second");

    g_ratiox=float(size.width/1024.0);
    g_ratioy=float(size.height/512.0);

    DX::ThrowIfFailed(
        m_dwriteFactory->CreateTextLayout(
        text->Data(),
        text->Length(),
        m_textFormat.Get(),
        1000, // maxWidth
        1000, // maxHeight
        &m_textLayout
        )
        );

    m_textLayout->GetMetrics(&m_textMetrics);
#endif

    m_d2dContext->BeginDraw();

    if(g_pImg)m_opacityBitmap->CopyFromMemory( NULL,(BYTE*)g_pImg, sizeof(unsigned int)*g_sizex );

    m_d2dContext->DrawBitmap( m_opacityBitmap.Get(), D2D1::RectF(0,0,size.width,size.height) );

#if SHOW_TEXT
    m_d2dContext->DrawTextLayout(
        D2D1::Point2F(0.0f, 0.0f),
        m_textLayout.Get(),
        m_Brush.Get(),
        D2D1_DRAW_TEXT_OPTIONS_CLIP
        );
#endif

    HRESULT hr = m_d2dContext->EndDraw();

    if (hr == D2DERR_RECREATE_TARGET){
        m_d2dContext->SetTarget(nullptr);
        m_d2dTargetBitmap = nullptr;
        CreateWindowSizeDependentResources();
    }else{
        DX::ThrowIfFailed(hr);
    }

    m_renderNeeded = false;
}

