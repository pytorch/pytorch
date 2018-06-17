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

#include "video.h"
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <pthread.h>

unsigned int* g_pImg = 0;
int g_sizex=0, g_sizey=0;
static video *g_video = 0;
static int g_fps = 0;
char *window_title=NULL;
#define WINDOW_TITLE_SIZE 256
int cocoa_update=0;

#include <sched.h>
#include <sys/time.h>
struct timeval g_time;

video::video()
#if __TBB_IOS
    : depth(24), red_shift(0), green_shift(8), blue_shift(16),
    red_mask(0xff), green_mask(0xff00), blue_mask(0xff0000)
#else
    : depth(24), red_shift(16), green_shift(8), blue_shift(0),
    red_mask(0xff0000), green_mask(0xff00), blue_mask(0xff)
#endif
{
    assert(g_video == 0);
    g_video = this; title = "Video"; cocoa_update=1; updating = true; calc_fps = false;
}

bool video::init_window(int x, int y)
{
    g_sizex = x; g_sizey = y;
    g_pImg = new unsigned int[x*y];
    if( window_title==NULL )
        window_title = (char*)malloc(WINDOW_TITLE_SIZE);
    strncpy( window_title, title, WINDOW_TITLE_SIZE-1 );
    running = true;
    return true;
}

bool video::init_console()
{
    running = true;
    return true;
}

void video::terminate()
{
    if(calc_fps) {
        double fps = g_fps;
        struct timezone tz; struct timeval end_time; gettimeofday(&end_time, &tz);
        fps /= (end_time.tv_sec+1.0*end_time.tv_usec/1000000.0) - (g_time.tv_sec+1.0*g_time.tv_usec/1000000.0);
        printf("%s: %.1f fps\n", title, fps);
    }
    g_video = 0; running = false;
    if(g_pImg) { delete[] g_pImg; g_pImg = 0; }
}

video::~video()
{
    if(g_video) terminate();
}

//! Count and display FPS count in titlebar
bool video::next_frame()
{
    if(calc_fps){
        if(!g_fps) {
            struct timezone tz; gettimeofday(&g_time, &tz);
        }
        g_fps++;
    }
    struct timezone tz; struct timeval now_time; gettimeofday(&now_time, &tz);
    double sec=((now_time.tv_sec+1.0*now_time.tv_usec/1000000.0) - (g_time.tv_sec+1.0*g_time.tv_usec/1000000.0));
    if( sec>1 ){
        if(calc_fps) {
            memcpy(&g_time, &now_time, sizeof(g_time));
            int fps;
            fps = g_fps/sec;
            cocoa_update = (int)updating;
            snprintf(window_title,WINDOW_TITLE_SIZE, "%s%s: %d fps", title, updating?"":" (no updating)", int(fps));
            g_fps=0;
        }
    }
    return running;
}


void* thread_func(void*)
{
    g_video->on_process();
    exit(EXIT_SUCCESS);
}

extern "C" void on_mouse_func(int x, int y, int k)
{
    g_video->on_mouse(x, y, k);
    return;
}

extern "C" void on_key_func(int x)
{
    g_video->on_key(x);
    return;
}

extern "C" int cocoa_main( int argc, char *argv[] );
//! Do standard loop
void video::main_loop()
{
    pthread_t handle;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_create(&handle,&attr,&thread_func,(void*)NULL);
    pthread_detach(handle);
    cocoa_main( 0, NULL );
}

//! Change window title
void video::show_title()
{
    if(title)
        strncpy( window_title, title, WINDOW_TITLE_SIZE );
    return;
}

///////////////////////////////////////////// public methods of video class ///////////////////////

drawing_area::drawing_area(int x, int y, int sizex, int sizey)
    : base_index(y*g_sizex + x), max_index(g_sizex*g_sizey), index_stride(g_sizex),
    pixel_depth(24), ptr32(g_pImg), start_x(x), start_y(y), size_x(sizex), size_y(sizey)
{
    assert(x < g_sizex); assert(y < g_sizey);
    assert(x+sizex <= g_sizex); assert(y+sizey <= g_sizey);

    index = base_index; // current index
}

void drawing_area::update()
{
    //nothing to do, updating via timer in cocoa part.
}
