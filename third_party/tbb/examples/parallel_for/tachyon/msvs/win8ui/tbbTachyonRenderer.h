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

#include <wrl.h>
#include "DirectXBase.h"

ref class tbbTachyonRenderer sealed : public DirectXBase
{
public:
    tbbTachyonRenderer();
    virtual void CreateDeviceIndependentResources() override;
    virtual void CreateDeviceResources() override;
    virtual void CreateWindowSizeDependentResources() override;
    virtual void Render() override;
    void Update(float timeTotal, float timeDelta);

    void UpdateView(Windows::Foundation::Point deltaViewPosition);

private:
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> m_Brush;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_textFormat;
    Microsoft::WRL::ComPtr<ID2D1Bitmap1> m_opacityBitmap;
    Microsoft::WRL::ComPtr<IDWriteTextLayout> m_textLayout;
    DWRITE_TEXT_METRICS m_textMetrics;
    bool m_renderNeeded;
    ~tbbTachyonRenderer();
};
