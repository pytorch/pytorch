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

#ifndef SEISMIC_VIDEO_H_
#define SEISMIC_VIDEO_H_

#include "../../common/gui/video.h"

class Universe;

class SeismicVideo : public video
{
#ifdef _WINDOWS
    #define MAX_LOADSTRING 100
    TCHAR szWindowClass[MAX_LOADSTRING];    // the main window class name
    WNDCLASSEX wcex;
#endif
    static const char * const titles[2];

    bool initIsParallel ;

    Universe &u_;
    int numberOfFrames_; // 0 means forever, positive means number of frames, negative is undefined
    int threadsHigh;
private:
    void on_mouse(int x, int y, int key);
    void on_process();

#ifdef _WINDOWS
public:
#endif
    void on_key(int key);

public:
    SeismicVideo(    Universe &u,int numberOfFrames, int threadsHigh, bool initIsParallel=true);
};
#endif /* SEISMIC_VIDEO_H_ */
