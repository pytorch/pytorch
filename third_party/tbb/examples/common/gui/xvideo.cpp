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

// Uncomment next line to disable shared memory features if you do not have libXext
// (http://www.xfree86.org/current/mit-shm.html)
//#define X_NOSHMEM

// Note that it may happen that the build environment supports the shared-memory extension
// (so there's no build-time reason to disable the relevant code by defining X_NOSHMEM),
// but that using shared memory still fails at run time.
// This situation will (ultimately) cause the error handler set by XSetErrorHandler()
// to be invoked with XErrorEvent::minor_code==X_ShmAttach. The code below tries to make
// such a determination at XShmAttach() time, which seems plausible, but unfortunately
// it has also been observed in a specific environment that the error may be reported
// at a later time instead, even after video::init_window() has returned.
// It is not clear whether this may happen in that way in any environment where it might
// depend on the kind of display, e.g., local vs. over "ssh -X", so #define'ing X_NOSHMEM
// may not always be the appropriate solution, therefore an environment variable
// has been introduced to disable shared memory at run time.
// A diagnostic has been added to advise the user about possible workarounds.
// X_ShmAttach macro was changed to 1 due to recent changes to X11/extensions/XShm.h header.

#include "video.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>
#include <signal.h>
#include <pthread.h>

#ifndef X_NOSHMEM
#include <errno.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>

static XShmSegmentInfo shmseginfo;
static Pixmap pixmap = 0;
static bool already_called_X_ShmAttach = false;
static bool already_advised_about_NOSHMEM_workarounds = false;
static const char* NOSHMEM_env_var_name = "TBB_EXAMPLES_X_NOSHMEM";
#endif
static char *display_name = NULL;
static Display *dpy = NULL;
static Screen *scrn;
static Visual *vis;
static Colormap cmap;
static GC gc;
static Window win, rootW;
static int dispdepth = 0;
static XGCValues xgcv;
static XImage *ximage;
static int x_error = 0;
static int vidtype = 3;
int g_sizex, g_sizey;
static video *g_video = 0;
unsigned int *g_pImg = 0;
static int g_fps = 0;
struct timeval g_time;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
Atom _XA_WM_DELETE_WINDOW = 0;// like in Xatom.h

///////////////////////////////////////////// public methods of video class ///////////////////////

video::video()
{
    assert(g_video == 0);
    g_video = this; title = "Video"; calc_fps = running = false; updating = true;
}

inline void mask2bits(unsigned int mask, unsigned int &save, depth_t &shift)
{
    save  = mask; if(!mask) { shift = dispdepth/3; return; }
    shift = 0; while(!(mask&1)) ++shift, mask >>= 1;
    int bits = 0; while(mask&1) ++bits,  mask >>= 1;
    shift += bits - 8;
}

int xerr_handler(Display* dpy_, XErrorEvent *error)
{
    x_error = error->error_code;
    if(g_video) g_video->running = false;
#ifndef X_NOSHMEM
    if (error->minor_code==1/*X_ShmAttach*/ && already_called_X_ShmAttach && !already_advised_about_NOSHMEM_workarounds)
    {
        char err[256]; XGetErrorText(dpy_, x_error, err, 255);
        fprintf(stderr, "Warning: Can't attach shared memory to display: %s (%d)\n", err, x_error);
        fprintf(stderr, "If you are seeing a black output window, try setting %s environment variable to 1"
                        " to disable shared memory extensions (0 to re-enable, other values undefined),"
                        " or rebuilding with X_NOSHMEM defined in " __FILE__ "\n", NOSHMEM_env_var_name);
        already_advised_about_NOSHMEM_workarounds = true;
    }
#else
    (void) dpy_; // warning prevention
#endif
    return 0;
}

bool video::init_window(int xsize, int ysize)
{
    { //enclose local variables before fail label
    g_sizex = xsize; g_sizey = ysize;

    // Open the display
    if (!dpy) {
        dpy = XOpenDisplay(display_name);
        if (!dpy) {
            fprintf(stderr, "Can't open X11 display %s\n", XDisplayName(display_name));
            goto fail;
        }
    }
    int theScreen = DefaultScreen(dpy);
    scrn = ScreenOfDisplay(dpy, theScreen);
    dispdepth = DefaultDepth(dpy, theScreen);
    XVisualInfo vinfo;
    if (!( (dispdepth >= 15 && dispdepth <= 32 && XMatchVisualInfo(dpy, theScreen, dispdepth, TrueColor, &vinfo) )
        || XMatchVisualInfo(dpy, theScreen, 24, TrueColor, &vinfo)
        || XMatchVisualInfo(dpy, theScreen, 32, TrueColor, &vinfo)
        || XMatchVisualInfo(dpy, theScreen, 16, TrueColor, &vinfo)
        || XMatchVisualInfo(dpy, theScreen, 15, TrueColor, &vinfo)
        )) {
        fprintf(stderr, "Display has no appropriate True Color visual\n");
        goto fail;
    }
    vis = vinfo.visual;
    depth = dispdepth = vinfo.depth;
    mask2bits(vinfo.red_mask, red_mask, red_shift);
    mask2bits(vinfo.green_mask, green_mask, green_shift);
    mask2bits(vinfo.blue_mask, blue_mask, blue_shift);
    rootW = RootWindow(dpy, theScreen);
    cmap = XCreateColormap(dpy, rootW, vis, AllocNone);
    XSetWindowAttributes attrs;
    attrs.backing_store = Always;
    attrs.colormap = cmap;
    attrs.event_mask = StructureNotifyMask|KeyPressMask|ButtonPressMask|ButtonReleaseMask;
    attrs.background_pixel = BlackPixelOfScreen(scrn);
    attrs.border_pixel = WhitePixelOfScreen(scrn);
    win = XCreateWindow(dpy, rootW,
        0, 0, xsize, ysize, 2,
        dispdepth, InputOutput, vis,
        CWBackingStore | CWColormap | CWEventMask |
        CWBackPixel | CWBorderPixel,
        &attrs);
    if(!win) {
        fprintf(stderr, "Can't create the window\n");
        goto fail;
    }
    XSizeHints sh;
    sh.flags = PSize | PMinSize | PMaxSize;
    sh.width = sh.min_width = sh.max_width = xsize;
    sh.height = sh.min_height = sh.max_height = ysize;
    XSetStandardProperties( dpy, win, g_video->title, g_video->title, None, NULL, 0, &sh );
    _XA_WM_DELETE_WINDOW = XInternAtom(dpy, "WM_DELETE_WINDOW", false);
    XSetWMProtocols(dpy, win, &_XA_WM_DELETE_WINDOW, 1);
    gc = XCreateGC(dpy, win, 0L, &xgcv);
    XMapRaised(dpy, win);
    XFlush(dpy);
#ifdef X_FULLSYNC
    XSynchronize(dpy, true);
#endif
    XSetErrorHandler(xerr_handler);

    int imgbytes = xsize*ysize*(dispdepth<=16?2:4);
    const char *vidstr;
#ifndef X_NOSHMEM
    int major, minor, pixmaps;
    if(XShmQueryExtension(dpy) &&
       XShmQueryVersion(dpy, &major, &minor, &pixmaps))
    { // Shared memory
        if(NULL!=getenv(NOSHMEM_env_var_name) && 0!=strcmp("0",getenv(NOSHMEM_env_var_name))) {
            goto generic;
        }
        shmseginfo.shmid = shmget(IPC_PRIVATE, imgbytes, IPC_CREAT|0777);
        if(shmseginfo.shmid < 0) {
            fprintf(stderr, "Warning: Can't get shared memory: %s\n", strerror(errno));
            goto generic;
        }
        g_pImg = (unsigned int*)(shmseginfo.shmaddr = (char*)shmat(shmseginfo.shmid, 0, 0));
        if(g_pImg == (unsigned int*)-1) {
            fprintf(stderr, "Warning: Can't attach to shared memory: %s\n", strerror(errno));
            shmctl(shmseginfo.shmid, IPC_RMID, NULL);
            goto generic;
        }
        shmseginfo.readOnly = false;
        if(!XShmAttach(dpy, &shmseginfo) || x_error) {
            char err[256]; XGetErrorText(dpy, x_error, err, 255);
            fprintf(stderr, "Warning: Can't attach shared memory to display: %s (%d)\n", err, x_error);
            shmdt(shmseginfo.shmaddr); shmctl(shmseginfo.shmid, IPC_RMID, NULL);
            goto generic;
        }
        already_called_X_ShmAttach = true;

#ifndef X_NOSHMPIX
        if(pixmaps && XShmPixmapFormat(dpy) == ZPixmap)
        { // Pixmaps
            vidtype = 2; vidstr = "X11 shared memory pixmap";
            pixmap = XShmCreatePixmap(dpy, win, (char*)g_pImg, &shmseginfo, xsize, ysize, dispdepth);
            XSetWindowBackgroundPixmap(dpy, win, pixmap);
        } else
#endif//!X_NOSHMPIX
        { // Standard
            vidtype = 1; vidstr = "X11 shared memory";
            ximage = XShmCreateImage(dpy, vis, dispdepth,
                ZPixmap, 0, &shmseginfo, xsize, ysize);
            if(!ximage) {
                fprintf(stderr, "Can't create the shared image\n");
                goto fail;
            }
            assert(ximage->bytes_per_line == xsize*(dispdepth<=16?2:4));
            ximage->data = shmseginfo.shmaddr;
        }
    } else
#endif
    {
#ifndef X_NOSHMEM
generic:
#endif
        vidtype = 0; vidstr = "generic X11";
        g_pImg = new unsigned int[imgbytes/sizeof(int)];
        ximage = XCreateImage(dpy, vis, dispdepth, ZPixmap, 0, (char*)g_pImg, xsize, ysize, 32, imgbytes/ysize);
        if(!ximage) {
            fprintf(stderr, "Can't create the image\n");
            goto fail;
        }
    }
    if( ximage ) {
        // Note: It may be more efficient to adopt the server's byte order
        //       and swap once per get_color() call instead of once per pixel.
        const uint32_t probe = 0x03020100;
        const bool big_endian = (((const char*)(&probe))[0]==0x03);
        ximage->byte_order = big_endian ? MSBFirst : LSBFirst;
    }
    printf("Note: using %s with %s visual for %d-bit color depth\n", vidstr, vis==DefaultVisual(dpy, theScreen)?"default":"non-default", dispdepth);
    running = true;
    return true;
    } // end of enclosing local variables
fail:
    terminate(); init_console();
    return false;
}

bool video::init_console()
{
    if(!g_pImg && g_sizex && g_sizey) {
        dispdepth = 24; red_shift = 16; vidtype = 3; // fake video
        g_pImg = new unsigned int[g_sizex*g_sizey];
        running = true;
    }
    return true;
}

void video::terminate()
{
    running = false;
    if(dpy) {
        vidtype = 3; // stop video
        if(threaded) { pthread_mutex_lock(&g_mutex); pthread_mutex_unlock(&g_mutex); }
        if(ximage) { XDestroyImage(ximage); ximage = 0; g_pImg = 0; } // it frees g_pImg for vidtype == 0
#ifndef X_NOSHMEM
        if(pixmap) XFreePixmap(dpy, pixmap);
        if(shmseginfo.shmaddr) { XShmDetach(dpy, &shmseginfo); shmdt(shmseginfo.shmaddr); g_pImg = 0; }
        if(shmseginfo.shmid >= 0) shmctl(shmseginfo.shmid, IPC_RMID, NULL);
#endif
        if(gc) XFreeGC(dpy, gc);
        if(win) XDestroyWindow(dpy, win);
        XCloseDisplay(dpy); dpy = 0;
    }
    if(g_pImg) { delete[] g_pImg; g_pImg = 0; } // if was allocated for console mode
}

video::~video()
{
    if(g_video) terminate();
    g_video = 0;
}

//! Do standard event loop
void video::main_loop()
{
    struct timezone tz; gettimeofday(&g_time, &tz);
    on_process();
}

//! Check for pending events once
bool video::next_frame()
{
    if(!running) return false;
    //! try acquire mutex if threaded code, returns on failure
    if(vidtype == 3 || threaded && pthread_mutex_trylock(&g_mutex))
        return running;
    //! Refresh screen picture
    g_fps++;
#ifndef X_NOSHMPIX
    if(vidtype == 2 && updating) XClearWindow(dpy, win);
#endif
    while( XPending(dpy) ) {
        XEvent report; XNextEvent(dpy, &report);
        switch( report.type ) {
            case ClientMessage:
                if(report.xclient.format != 32 || report.xclient.data.l[0] != _XA_WM_DELETE_WINDOW) break;
            case DestroyNotify:
                running = false;
            case KeyPress:
                on_key( XLookupKeysym(&report.xkey, 0) ); break;
            case ButtonPress:
                on_mouse( report.xbutton.x, report.xbutton.y, report.xbutton.button ); break;
            case ButtonRelease:
                on_mouse( report.xbutton.x, report.xbutton.y, -report.xbutton.button ); break;
        }
    }
    struct timezone tz; struct timeval now_time; gettimeofday(&now_time, &tz);
    double sec = (now_time.tv_sec+1.0*now_time.tv_usec/1000000.0) - (g_time.tv_sec+1.0*g_time.tv_usec/1000000.0);
    if(sec > 1) {
        memcpy(&g_time, &now_time, sizeof(g_time));
        if(calc_fps) {
            double fps = g_fps; g_fps = 0;
            char buffer[256]; snprintf(buffer, 256, "%s%s: %d fps", title, updating?"":" (no updating)", int(fps/sec));
            XStoreName(dpy, win, buffer);
        }
#ifndef X_FULLSYNC
        XSync(dpy, false); // It is often better then using XSynchronize(dpy, true)
#endif//X_FULLSYNC
    }
    if(threaded) pthread_mutex_unlock(&g_mutex);
    return true;
}

//! Change window title
void video::show_title()
{
    if(vidtype < 3)
        XStoreName(dpy, win, title);
}

drawing_area::drawing_area(int x, int y, int sizex, int sizey)
    : base_index(y*g_sizex + x), max_index(g_sizex*g_sizey), index_stride(g_sizex),
    pixel_depth(dispdepth), ptr32(g_pImg), start_x(x), start_y(y), size_x(sizex), size_y(sizey)
{
    assert(x < g_sizex); assert(y < g_sizey);
    assert(x+sizex <= g_sizex); assert(y+sizey <= g_sizey);

    index = base_index; // current index
}

void drawing_area::update()
{
    if(!g_video->updating) return;
#ifndef X_NOSHMEM
    switch(vidtype) {
    case 0:
#endif
        pthread_mutex_lock(&g_mutex);
        if(vidtype == 0) XPutImage(dpy, win, gc, ximage, start_x, start_y, start_x, start_y, size_x, size_y);
        pthread_mutex_unlock(&g_mutex);
#ifndef X_NOSHMEM
        break;
    case 1:
        pthread_mutex_lock(&g_mutex);
        if(vidtype == 1) XShmPutImage(dpy, win, gc, ximage, start_x, start_y, start_x, start_y, size_x, size_y, false);
        pthread_mutex_unlock(&g_mutex);
        break;
    /*case 2: make it in next_frame(); break;*/
    }
#endif
}
