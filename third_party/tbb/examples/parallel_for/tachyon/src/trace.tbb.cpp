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

/*
    The original source for this example is
    Copyright (c) 1994-2008 John E. Stone
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. The name of the author may not be used to endorse or promote products
       derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
*/

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "tgafile.h"
#include "trace.h"
#include "light.h"
#include "shade.h"
#include "camera.h"
#include "util.h"
#include "intersect.h"
#include "global.h"
#include "ui.h"
#include "tachyon_video.h"

// shared but read-only so could be private too
static thr_parms *all_parms;
static scenedef scene;
static int startx;
static int stopx;
static int starty;
static int stopy;
static flt jitterscale;
static int totaly;

#ifdef MARK_RENDERING_AREA

// rgb colors list for coloring image by each thread
static const float inner_alpha = 0.3;
static const float border_alpha = 0.5;
#define NUM_COLORS 24
static int colors[NUM_COLORS][3] = {
    {255,110,0},    {220,254,0},    {102,254,0},    {0,21,254},     {97,0,254},     {254,30,0},
    {20,41,8},      {144,238,38},   {184,214,139},  {28,95,20},     {139,173,148},  {188,228,183},
    {145,47,56},    {204,147,193},  {45,202,143},   {204,171,143},  {143,160,204},  {220,173,3},
    {1,152,231},    {79,235,237},   {52,193,72},    {67,136,151},   {78,87,179},    {143,255,9},
};

#include "tbb/atomic.h"
#include "tbb/enumerable_thread_specific.h"
// storage and counter for thread numbers in order of first task run
typedef tbb::enumerable_thread_specific< int > thread_id_t;
thread_id_t thread_ids (-1);
tbb::atomic<int> thread_number;

#endif

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/spin_mutex.h"
#include "tbb/blocked_range2d.h"

static tbb::spin_mutex MyMutex, MyMutex2;

static color_t render_one_pixel (int x, int y, unsigned int *local_mbox, unsigned int &serial,
                                 int startx, int stopx, int starty, int stopy
#ifdef MARK_RENDERING_AREA
                                 , int *blend, float alpha
#endif
)
{
    /* private vars moved inside loop */
    ray primary, sample;
    color col, avcol;
    int R,G,B;
    intersectstruct local_intersections;    
    int alias;
    /* end private */

    primary=camray(&scene, x, y);
    primary.intstruct = &local_intersections;
    primary.flags = RT_RAY_REGULAR;

    serial++;
    primary.serial = serial;  
    primary.mbox = local_mbox;
    primary.maxdist = FHUGE;
    primary.scene = &scene;
    col=trace(&primary);  

    serial = primary.serial;

    /* perform antialiasing if enabled.. */
    if (scene.antialiasing > 0) {
        for (alias=0; alias < scene.antialiasing; alias++) {

            serial++; /* increment serial number */
            sample=primary;  /* copy the regular primary ray to start with */
            sample.serial = serial; 

            {
                tbb::spin_mutex::scoped_lock lock (MyMutex);
                sample.d.x+=((rand() % 100) - 50) / jitterscale;
                sample.d.y+=((rand() % 100) - 50) / jitterscale;
                sample.d.z+=((rand() % 100) - 50) / jitterscale;
            }

            avcol=trace(&sample);  

            serial = sample.serial; /* update our overall serial # */

            col.r += avcol.r;
            col.g += avcol.g;
            col.b += avcol.b;
        }

        col.r /= (scene.antialiasing + 1.0);
        col.g /= (scene.antialiasing + 1.0);
        col.b /= (scene.antialiasing + 1.0);
    }

    /* Handle overexposure and underexposure here... */
    R=(int) (col.r*255);
    if (R > 255) R = 255;
    else if (R < 0) R = 0;

    G=(int) (col.g*255);
    if (G > 255) G = 255;
    else if (G < 0) G = 0;

    B=(int) (col.b*255);
    if (B > 255) B = 255;
    else if (B < 0) B = 0;

#ifdef MARK_RENDERING_AREA
    R = int((1.0 - alpha) * R + alpha * blend[0]);
    G = int((1.0 - alpha) * G + alpha * blend[1]);
    B = int((1.0 - alpha) * B + alpha * blend[2]);
#endif
    
    return video->get_color(R, G, B);
}

class parallel_task {
public:
    void operator() (const tbb::blocked_range2d<int> &r) const
    {
       // task-local storage
        unsigned int serial = 1;
        unsigned int mboxsize = sizeof(unsigned int)*(max_objectid() + 20);
        unsigned int * local_mbox = (unsigned int *) alloca(mboxsize);
        memset(local_mbox,0,mboxsize);
#ifdef MARK_RENDERING_AREA
        // compute thread number while first task run
        thread_id_t::reference thread_id = thread_ids.local();
        if (thread_id == -1) thread_id = thread_number++;
        // choose thread color
        int pos = thread_id % NUM_COLORS;
        if(video->running) {
            drawing_area drawing(r.cols().begin(), totaly-r.rows().end(), r.cols().end() - r.cols().begin(), r.rows().end()-r.rows().begin());
            for (int i = 1, y = r.rows().begin(); y != r.rows().end(); ++y, i++) {
                drawing.set_pos(0, drawing.size_y-i);
                for (int x = r.cols().begin(); x != r.cols().end(); x++) {
                    int d = (y % 3 == 0) ? 2 : 1;
                    drawing.put_pixel(video->get_color(colors[pos][0]/d, colors[pos][1]/d, colors[pos][2]/d));
                }
            }
        }
#endif
        if(video->next_frame()) {
            drawing_area drawing(r.cols().begin(), totaly-r.rows().end(), r.cols().end() - r.cols().begin(), r.rows().end()-r.rows().begin());
            for (int i = 1, y = r.rows().begin(); y != r.rows().end(); ++y, i++) {
                drawing.set_pos(0, drawing.size_y-i);
                for (int x = r.cols().begin(); x != r.cols().end(); x++) {
#ifdef MARK_RENDERING_AREA
                    float alpha = y==r.rows().begin()||y==r.rows().end()-1||x==r.cols().begin()||x==r.cols().end()-1
                                ? border_alpha : inner_alpha;
                    color_t c = render_one_pixel (x, y, local_mbox, serial, startx, stopx, starty, stopy, colors[pos], alpha);
#else
                    color_t c = render_one_pixel (x, y, local_mbox, serial, startx, stopx, starty, stopy);
#endif
                    drawing.put_pixel(c);
                }
            }
        }
    }

    parallel_task () {}
};

void * thread_trace(thr_parms * parms)
{
#if !WIN8UI_EXAMPLE
    int n, nthreads = tbb::task_scheduler_init::automatic;
    char *nthreads_str = getenv ("TBB_NUM_THREADS");
    if (nthreads_str && (sscanf (nthreads_str, "%d", &n) > 0) && (n > 0)) nthreads = n;
    tbb::task_scheduler_init init (nthreads);
#endif

    // shared but read-only so could be private too
    all_parms = parms;
    scene = parms->scene;
    startx = parms->startx;
    stopx = parms->stopx;
    starty = parms->starty;
    stopy = parms->stopy;
    jitterscale = 40.0*(scene.hres + scene.vres);
    totaly = parms->scene.vres;
#ifdef MARK_RENDERING_AREA
    thread_ids.clear();
#endif

    int grain_size = 8;
//WIN8UI does not support getenv() function so using auto_partitioner unconditionally
#if !WIN8UI_EXAMPLE
    int g;
    char *grain_str = getenv ("TBB_GRAINSIZE");
    if (grain_str && (sscanf (grain_str, "%d", &g) > 0) && (g > 0)) grain_size = g;
    char *sched_str = getenv ("TBB_PARTITIONER");
    static tbb::affinity_partitioner g_ap; // reused across calls to thread_trace
    if ( sched_str && !strncmp(sched_str, "aff", 3) )
        tbb::parallel_for (tbb::blocked_range2d<int> (starty, stopy, grain_size, startx, stopx, grain_size), parallel_task (), g_ap);
    else if ( sched_str && !strncmp(sched_str, "simp", 4) )
        tbb::parallel_for (tbb::blocked_range2d<int> (starty, stopy, grain_size, startx, stopx, grain_size), parallel_task (), tbb::simple_partitioner());
    else
#endif
        tbb::parallel_for (tbb::blocked_range2d<int> (starty, stopy, grain_size, startx, stopx, grain_size), parallel_task (), tbb::auto_partitioner());

    return(NULL);  
}
