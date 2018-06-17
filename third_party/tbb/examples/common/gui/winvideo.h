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

/////// Common internal implementation of Windows-specific stuff //////////////
///////                  Must be the first included header       //////////////

#ifndef __WINVIDEO_H__
#define __WINVIDEO_H__

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
// Check that the target Windows version has all API calls requried.
#ifndef _WIN32_WINNT
# define _WIN32_WINNT 0x0400
#endif
#if _WIN32_WINNT<0x0400
# define YIELD_TO_THREAD() Sleep(0)
#else
# define YIELD_TO_THREAD() SwitchToThread()
#endif
#include "video.h"
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

// maximum mumber of lines the output console should have
static const WORD MAX_CONSOLE_LINES = 500;
const COLORREF              RGBKEY = RGB(8, 8, 16); // at least 8 for 16-bit palette
HWND                        g_hAppWnd;           // The program's window handle
HANDLE                      g_handles[2] = {0,0};// thread and wake up event
unsigned int *              g_pImg = 0;          // drawing memory
int                         g_sizex, g_sizey;
static video *              g_video = 0;
WNDPROC                     g_pUserProc = 0;
HINSTANCE                   video::win_hInstance = 0;
int                         video::win_iCmdShow = 0;
static WNDCLASSEX *         gWndClass = 0;
static HACCEL               hAccelTable = 0;
static DWORD                g_msec = 0;
static int g_fps = 0, g_updates = 0, g_skips = 0;

bool DisplayError(LPSTR lpstrErr, HRESULT hres = 0); // always returns false
LRESULT CALLBACK InternalWndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

//! Create window
bool WinInit(HINSTANCE hInstance, int nCmdShow, WNDCLASSEX *uwc, const char *title, bool fixedsize)
{
    WNDCLASSEX wndclass;  // Our app's windows class
    if(uwc) {
        memcpy(&wndclass, uwc, sizeof(wndclass));
        g_pUserProc = uwc->lpfnWndProc;
    } else {
        memset(&wndclass, 0, sizeof(wndclass));
        wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
        wndclass.lpszClassName = title;
    }
    wndclass.cbSize = sizeof(wndclass);
    wndclass.hInstance = hInstance;
    wndclass.lpfnWndProc = InternalWndProc;
    wndclass.style |= CS_HREDRAW | CS_VREDRAW;
    wndclass.hbrBackground = CreateSolidBrush(RGBKEY);

    if( !RegisterClassExA(&wndclass) ) return false;
    int xaddend = GetSystemMetrics(fixedsize?SM_CXFIXEDFRAME:SM_CXFRAME)*2;
    int yaddend = GetSystemMetrics(fixedsize?SM_CYFIXEDFRAME:SM_CYFRAME)*2 + GetSystemMetrics(SM_CYCAPTION);
    if(wndclass.lpszMenuName) yaddend += GetSystemMetrics(SM_CYMENU);

    // Setup the new window's physical parameters - and tell Windows to create it
    g_hAppWnd = CreateWindowA(wndclass.lpszClassName,  // Window class name
                             title,  // Window caption
                             !fixedsize ? WS_OVERLAPPEDWINDOW :  // Window style
                             WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX,
                             CW_USEDEFAULT,  // Initial x pos: use default placement
                             0,              // Initial y pos: not used here
                             g_sizex+xaddend,// Initial x size
                             g_sizey+yaddend,// Initial y size
                             NULL,      // parent window handle
                             NULL,      // window menu handle
                             hInstance, // program instance handle
                             NULL);     // Creation parameters
    return g_hAppWnd != NULL;
}

//! create console window with redirection
static bool RedirectIOToConsole(void)
{
    int hConHandle; size_t lStdHandle;
    CONSOLE_SCREEN_BUFFER_INFO coninfo;
    FILE *fp;
    // allocate a console for this app
    AllocConsole();

    // set the screen buffer to be big enough to let us scroll text
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &coninfo);
    coninfo.dwSize.Y = MAX_CONSOLE_LINES;
    SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), coninfo.dwSize);

    // redirect unbuffered STDOUT to the console
    lStdHandle = (size_t)GetStdHandle(STD_OUTPUT_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    if(hConHandle <= 0) return false;
    fp = _fdopen( hConHandle, "w" );
    *stdout = *fp;
    setvbuf( stdout, NULL, _IONBF, 0 );

    // redirect unbuffered STDERR to the console
    lStdHandle = (size_t)GetStdHandle(STD_ERROR_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    if(hConHandle > 0) {
        fp = _fdopen( hConHandle, "w" );
        *stderr = *fp;
        setvbuf( stderr, NULL, _IONBF, 0 );
    }

    // redirect unbuffered STDIN to the console
    lStdHandle = (size_t)GetStdHandle(STD_INPUT_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    if(hConHandle > 0) {
        fp = _fdopen( hConHandle, "r" );
        *stdin = *fp;
        setvbuf( stdin, NULL, _IONBF, 0 );
    }

    // make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog
    // point to console as well
    std::ios::sync_with_stdio();
    return true;
}


video::video()
    : depth(24), red_shift(16), green_shift(8), blue_shift(0),
      red_mask(0xff0000), green_mask(0xff00), blue_mask(0xff)
{
    assert(g_video == 0);
    g_video = this; title = "Video"; running = threaded = calc_fps = false; updating = true;
}

//! optionally call it just before init() to set own
void video::win_set_class(WNDCLASSEX &wcex)
{
    gWndClass = &wcex;
}

void video::win_load_accelerators(int idc)
{
    hAccelTable = LoadAccelerators(win_hInstance, MAKEINTRESOURCE(idc));
}

bool video::init_console()
{
    if(RedirectIOToConsole()) {
        if(!g_pImg && g_sizex && g_sizey)
            g_pImg = new unsigned int[g_sizex * g_sizey];
        if(g_pImg) running = true;
        return true;
    }
    return false;
}

video::~video()
{
    if(g_video) terminate();
}

DWORD WINAPI thread_video(LPVOID lpParameter)
{
    video *v = (video*)lpParameter;
    v->on_process();
    return 0;
}

static bool loop_once(video *v)
{
    // screen update notify
    if(int updates = g_updates) {
        g_updates = 0;
        if(g_video->updating) { g_skips += updates-1; g_fps++; }
        else g_skips += updates;
        UpdateWindow(g_hAppWnd);
    }
    // update fps
    DWORD msec = GetTickCount();
    if(v->calc_fps && msec >= g_msec+1000) {
        double sec = (msec - g_msec)/1000.0;
        char buffer[256], n = _snprintf(buffer, 128, "%s: %d fps", v->title, int(double(g_fps + g_skips)/sec));
        if(g_skips) _snprintf(buffer+n, 128, " - %d skipped = %d updates", int(g_skips/sec), int(g_fps/sec));
        SetWindowTextA(g_hAppWnd, buffer);
        g_msec = msec; g_skips = g_fps = 0;
    }
    // event processing, including painting
    MSG msg;
    if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
        if( msg.message == WM_QUIT ) { v->running = false; return false; }
        if( !hAccelTable || !TranslateAccelerator(msg.hwnd, hAccelTable, &msg) ){
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        return true; // try again
    }
    return false;
}

//! Do standard event loop
void video::main_loop()
{
    // let Windows draw and unroll the window
    InvalidateRect(g_hAppWnd, 0, false);
    g_msec = GetTickCount(); // let's stay for 0,5 sec
    while(g_msec + 500 > GetTickCount()) { loop_once(this); Sleep(1); }
    g_msec = GetTickCount();
    // now, start main process
    if(threaded) {
        g_handles[0] = CreateThread (
            NULL,             // LPSECURITY_ATTRIBUTES security_attrs
            0,                // SIZE_T stacksize
            (LPTHREAD_START_ROUTINE) thread_video,
            this,               // argument
            0, 0);
        if(!g_handles[0]) { DisplayError("Can't create thread"); return; }
        else // harmless race is possible here
            g_handles[1] = CreateEvent(NULL, false, false, NULL);
        while(running) {
            while(loop_once(this));
            YIELD_TO_THREAD(); // give time for processing when running on single CPU
            DWORD r = MsgWaitForMultipleObjects(2, g_handles, false, INFINITE, QS_ALLINPUT^QS_MOUSEMOVE);
            if(r == WAIT_OBJECT_0) break; // thread terminated
        }
        running = false;
        if(WaitForSingleObject(g_handles[0], 3000) == WAIT_TIMEOUT){
            // there was not enough time for graceful shutdown, killing the example with code 1.
            exit(1);
        }
        if(g_handles[0]) CloseHandle(g_handles[0]);
        if(g_handles[1]) CloseHandle(g_handles[1]);
        g_handles[0] = g_handles[1] = 0;
    }
    else on_process();
}

//! Refresh screen picture
bool video::next_frame()
{
    if(!running) return false;
    g_updates++; // Fast but inaccurate counter. The data race here is benign.
    if(!threaded) while(loop_once(this));
    else if(g_handles[1]) {
        SetEvent(g_handles[1]);
        YIELD_TO_THREAD();
    }
    return true;
}

//! Change window title
void video::show_title()
{
    if(g_hAppWnd)
        SetWindowTextA(g_hAppWnd, title);
}

#endif //__WINVIDEO_H__
