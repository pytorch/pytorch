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

#ifndef __VIDEO_H__
#define __VIDEO_H__

#include <cassert>
#if _MSC_VER
#include <stddef.h> // for uintptr_t
#else
#include <stdint.h> // for uintptr_t
#endif
#if _WIN32 || _WIN64
#include <windows.h>
#else
#include <unistd.h>
#endif

typedef unsigned int color_t;
typedef unsigned char colorcomp_t;
typedef signed char depth_t;

//! Class for getting access to drawing memory
class drawing_memory
{
#ifdef __TBB_MIC_OFFLOAD
    // The address is kept as uintptr_t since
    // the compiler could not offload a pointer
#endif
    uintptr_t   my_address;
public:
    depth_t     pixel_depth;
    int         sizex, sizey;
    //! Get drawing memory
    inline char* get_address() const { return reinterpret_cast<char*>(my_address); }
    //! Get drawing memory size
    inline int get_size() const { return ((pixel_depth>16) ? 4:2) * sizex * sizey; }
    //! Set drawing memory
    inline void set_address(char *mem) { my_address = reinterpret_cast<uintptr_t>(mem); }

    friend class drawing_area;
    friend class video;
};

//! Simple proxy class for managing of different video systems
class video
{
    //! colorspace information
    depth_t depth, red_shift, green_shift, blue_shift;
    color_t red_mask, green_mask, blue_mask;
    friend class drawing_area;

public:
    //! Constructor
    video();
    //! Destructor
    ~video();
    //! member to set window name
    const char *title;
    //! true is enable to show fps
    bool calc_fps;
    //! if true: on windows fork processing thread for on_process(), on non-windows note that next_frame() is called concurrently.
    bool threaded;
    //! true while running within main_loop()
    bool running;
    //! if true, do gui updating
    bool updating;
    //! initialize graphical video system
    bool init_window(int sizex, int sizey);
    //! initialize console. returns true if console is available
    bool init_console();
    //! terminate video system
    void terminate();
    //! Do standard event & processing loop. Use threaded = true to separate event/updating loop from frame processing
    void main_loop();
    //! Process next frame
    bool next_frame();
    //! Change window title
    void show_title();
    //! translate RGB components into packed type
    inline color_t get_color(colorcomp_t red, colorcomp_t green, colorcomp_t blue) const;
    //! Get drawing memory descriptor
    inline drawing_memory get_drawing_memory() const;

    //! code of the ESCape key
    static const int esc_key = 27;
    //! Mouse events handler.
    virtual void on_mouse(int x, int y, int key) { }
    //! Mouse events handler.
    virtual void on_key(int key) { }
    //! Main processing loop. Redefine with your own
    virtual void on_process() { while(next_frame()); }

#ifdef _WINDOWS
    //! Windows specific members
    //! if VIDEO_WINMAIN isn't defined then set this just before init() by arguments of WinMain
    static HINSTANCE win_hInstance; static int win_iCmdShow;
    //! optionally call it just before init() to set own. Use ascii strings convention
    void win_set_class(WNDCLASSEX &);
    //! load and set accelerator table from resources
    void win_load_accelerators(int idc);
#endif
};

//! Drawing class
class drawing_area
{
    const size_t base_index, max_index, index_stride;
    const depth_t pixel_depth;
    unsigned int * const ptr32;
    size_t index;
public:
    const int start_x, start_y, size_x, size_y;
    //! constructors
    drawing_area(int x, int y, int sizex, int sizey);
    inline drawing_area(int x, int y, int sizex, int sizey, const drawing_memory &dmem);
    //! destructor
    inline ~drawing_area();
    //! update the image
    void update();
    //! set current position. local_x could be bigger then size_x
    inline void set_pos(int local_x, int local_y);
    //! put pixel in current position with incremental address calculating to next right pixel
    inline void put_pixel(color_t color);
    //! draw pixel at position by packed color
    void set_pixel(int localx, int localy, color_t color)
        { set_pos(localx, localy); put_pixel(color); }
};

extern int g_sizex;
extern int g_sizey;
extern unsigned int *g_pImg;

inline drawing_memory video::get_drawing_memory() const
{
    drawing_memory dmem;
    dmem.pixel_depth = depth;
    dmem.my_address = reinterpret_cast<uintptr_t>(g_pImg);
    dmem.sizex = g_sizex;
    dmem.sizey = g_sizey;
    return dmem;
}

inline color_t video::get_color(colorcomp_t red, colorcomp_t green, colorcomp_t blue) const
{
    if(red_shift == 16) // only for depth == 24 && red_shift > blue_shift
        return (red<<16) | (green<<8) | blue;
    else if(depth >= 24)
        return
#if __ANDROID__
                // Setting Alpha to 0xFF
                0xFF000000 |
#endif
                (red<<red_shift) | (green<<green_shift) | (blue<<blue_shift);
    else if(depth > 0) {
        depth_t bs = blue_shift, rs = red_shift;
        if(blue_shift < 0) blue >>= -bs, bs = 0;
        else /*red_shift < 0*/ red >>= -rs, rs = 0;
        return ((red<<rs)&red_mask) | ((green<<green_shift)&green_mask) | ((blue<<bs)&blue_mask);
    } else { // UYVY colorspace
        unsigned y, u, v;
        y = red * 77 + green * 150 + blue * 29; // sum(77+150+29=256) * max(=255):  limit->2^16
        u = (2048 + (blue << 3) - (y >> 5)) >> 4; // (limit->2^12)>>4
        v = (2048 + (red << 3) - (y >> 5)) >> 4;
        y = y >> 8;
        return u | (y << 8) | (v << 16) | (y << 24);
    }
}

inline drawing_area::drawing_area(int x, int y, int sizex, int sizey, const drawing_memory &dmem)
    : base_index(y*dmem.sizex + x), max_index(dmem.sizex*dmem.sizey), index_stride(dmem.sizex),
    pixel_depth(dmem.pixel_depth), ptr32(reinterpret_cast<unsigned int*>(dmem.my_address)),
    start_x(x), start_y(y), size_x(sizex), size_y(sizey)
{
    assert(x < dmem.sizex); assert(y < dmem.sizey);
    assert(x+sizex <= dmem.sizex); assert(y+sizey <= dmem.sizey);

    index = base_index; // current index
}

inline void drawing_area::set_pos(int local_x, int local_y)
{
    index = base_index + local_x + local_y*index_stride;
}

inline void drawing_area::put_pixel(color_t color)
{
    assert(index < max_index);
    if(pixel_depth > 16) ptr32[index++] = color;
    else if(pixel_depth > 0)
        ((unsigned short*)ptr32)[index++] = (unsigned short)color;
    else { // UYVY colorspace
        if(index&1) color >>= 16;
        ((unsigned short*)ptr32)[index++] = (unsigned short)color;
    }
}

inline drawing_area::~drawing_area()
{
#if ! __TBB_DEFINE_MIC
    update();
#endif
}

#if defined(_WINDOWS) && (defined(VIDEO_WINMAIN) || defined(VIDEO_WINMAIN_ARGS) )
#include <cstdlib>
//! define WinMain for subsystem:windows.
#ifdef VIDEO_WINMAIN_ARGS
int main(int, char *[]);
#else
int main();
#endif
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR szCmdLine, int iCmdShow)
{
    video::win_hInstance = hInstance;  video::win_iCmdShow = iCmdShow;
#ifdef VIDEO_WINMAIN_ARGS
    return main(__argc, __argv);
#else
    return main();
#endif
}
#endif

#endif// __VIDEO_H__
