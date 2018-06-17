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

#include "DirectXPage.g.h"
#include "tbbTachyonRenderer.h"

namespace tbbTachyon
{
    [Windows::Foundation::Metadata::WebHostHidden]
    public ref class DirectXPage sealed
    {
    public:
        DirectXPage();

    private:
        ~DirectXPage();
        void OnRendering(Object^ sender, Object^ args);

        Windows::Foundation::EventRegistrationToken m_eventToken;

        tbbTachyonRenderer^ m_renderer;
        bool m_renderNeeded;
        int m_number_of_threads;

        void ThreadsSliderValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e);
        void ThreadsApply_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        void Exit_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    };
}
