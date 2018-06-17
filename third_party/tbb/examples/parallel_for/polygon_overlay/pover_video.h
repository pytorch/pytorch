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

// support for GUI for polygon overlay demo
//
#ifndef _POVER_VIDEO_H_
#define _POVER_VIDEO_H_
#include "../../common/gui/video.h"

#include "pover_global.h"  // for declaration of DEFINE and INIT

DEFINE class video *gVideo INIT(0);

DEFINE int n_next_frame_calls INIT(0);
DEFINE int frame_skips INIT(10);
extern bool g_next_frame();
extern bool g_last_frame();

class pover_video: public video {
    void on_process();
public:
#ifdef _WINDOWS
    bool graphic_display(){return video::win_hInstance != (HINSTANCE)NULL;}
#else
    bool graphic_display() { return true;} // fix this for Linux
#endif
    //void on_key(int key);
};

DEFINE int g_xwinsize INIT(1024);
DEFINE int g_ywinsize INIT(768);

DEFINE int map1XLoc INIT(10);
DEFINE int map1YLoc INIT(10);
DEFINE int map2XLoc INIT(270);
DEFINE int map2YLoc INIT(10);
DEFINE int maprXLoc INIT(530);
DEFINE int maprYLoc INIT(10);

DEFINE const char *g_windowTitle INIT("Polygon Overlay");
DEFINE bool g_useGraphics INIT(true);

extern bool initializeVideo(int argc, char **argv);

extern void rt_sleep(int msec);

#endif  // _POVER_VIDEO_H_
