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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VIDEO_WINMAIN_ARGS
#include "types.h"
#include "api.h"       /* The ray tracing library API */
#include "parse.h"     /* Support for my own file format */
#include "ui.h"
#include "util.h"
#include "tachyon_video.h"
#include "../../../common/utility/utility.h"

#if WIN8UI_EXAMPLE
#include "tbb/tbb.h"
volatile long global_startTime = 0;
volatile long global_elapsedTime = 0;
volatile bool global_isCancelled = false;
volatile int global_number_of_threads;
#endif

SceneHandle global_scene;
int global_xsize;     /*  size of graphic image rendered in window (from hres, vres)  */
int global_ysize;
int global_xwinsize;  /*  size of window (may be larger than above)  */
int global_ywinsize;
char *global_window_title;
bool global_usegraphics;

bool silent_mode = false; /* silent mode */

class tachyon_video *video = 0;

typedef struct {
  int foundfilename;      /* was a model file name found in the args? */
  char filename[1024];    /* model file to render */
  int useoutfilename;     /* command line override of output filename */
  char outfilename[1024]; /* name of output image file */
  int verbosemode;        /* verbose flags */
  int antialiasing;       /* antialiasing setting */
  int displaymode;        /* display mode */
  int boundmode;          /* bounding mode */
  int boundthresh;        /* bounding threshold */
  int usecamfile;         /* use camera file */
  char camfilename[1024]; /* camera filename */
} argoptions;

void initoptions(argoptions * opt) {
    memset(opt, 0, sizeof(argoptions));
    opt->foundfilename = -1;
    opt->useoutfilename = -1;
    opt->verbosemode = -1;
    opt->antialiasing = -1;
    opt->displaymode = -1;
    opt->boundmode = -1; 
    opt->boundthresh = -1; 
    opt->usecamfile = -1;
}

#if WIN8UI_EXAMPLE
int CreateScene() {

   char* filename = "Assets/balls.dat";

    global_scene = rt_newscene();
    rt_initialize();

    if ( readmodel(filename, global_scene) != 0 ) {
        rt_finalize();
        return -1;
    }

    // need these early for create_graphics_window() so grab these here...
    scenedef *scene = (scenedef *) global_scene;

    // scene->hres and scene->vres should be equal to screen resolution
    scene->hres = global_xwinsize = global_xsize;
    scene->vres = global_ywinsize = global_ysize;  

    return 0;
}

unsigned int __stdcall example_main(void *)
{
    try {

        if ( CreateScene() != 0 )
            exit(-1);

        tachyon_video tachyon;
        tachyon.threaded = true;
        tachyon.init_console();

        // always using window even if(!global_usegraphics)
        global_usegraphics = 
            tachyon.init_window(global_xwinsize, global_ywinsize);
        if(!tachyon.running)
            exit(-1);

        video = &tachyon;

        for(;;) {
            global_elapsedTime = 0;
            global_startTime=(long) time(NULL);
            global_isCancelled=false;
            if (video)video->running = true;
            tbb::task_scheduler_init init (global_number_of_threads);
            memset(g_pImg, 0, sizeof(unsigned int) * global_xsize * global_ysize);
            tachyon.main_loop();
            global_elapsedTime = (long)(time(NULL)-global_startTime);
            video->running=false;
            //The timer to restart drawing then it is complete.
            int timer=50;
            while( (  !global_isCancelled && (timer--)>0 ) ){
                rt_sleep( 100 );
            }
        }
        return NULL;

    } catch ( std::exception& e ) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}

#elif __TBB_IOS

#include "tbb/tbb.h"
#include "CoreFoundation/CoreFoundation.h"
extern "C" void get_screen_resolution(int *x, int *y);

int CreateScene() {

    CFURLRef balls_dat_url = CFBundleCopyResourceURL(CFBundleGetMainBundle(), CFSTR("balls"), CFSTR("dat"),NULL);
    char filename[1024];
    CFURLGetFileSystemRepresentation(balls_dat_url, true, (UInt8*)filename, (CFIndex)sizeof(filename));
    CFRelease(balls_dat_url);

    global_scene = rt_newscene();
    rt_initialize();

    if ( readmodel(filename, global_scene) != 0 ) {
        rt_finalize();
        return -1;
    }

    // need these early for create_graphics_window() so grab these here...
    scenedef *scene = (scenedef *) global_scene;

    get_screen_resolution(&global_xsize, &global_ysize);

    // scene->hres and scene->vres should be equal to screen resolution
    scene->hres = global_xwinsize = global_xsize;
    scene->vres = global_ywinsize = global_ysize;
    return 0;
}

int main (int argc, char *argv[]) {
    try {

        if ( CreateScene() != 0 ) return -1;

        tachyon_video tachyon;
        tachyon.threaded = true;
        tachyon.init_console();

        global_usegraphics = tachyon.init_window(global_xwinsize, global_ywinsize);
        if(!tachyon.running) return -1;

        //TODO: add a demo loop.
        video = &tachyon;
        if (video)video->running = true;
        memset(g_pImg, 0, sizeof(unsigned int) * global_xsize * global_ysize);
        tachyon.main_loop();
        video->running=false;
        return NULL;

    } catch ( std::exception& e ) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}

#else

static char *window_title_string (int argc, const char **argv)
{
    int i;
    char *name;

    name = (char *) malloc (8192);
    char *title = getenv ("TITLE");
    if( title ) strcpy( name, title );
    else {
        if(strrchr(argv[0], '\\')) strcpy (name, strrchr(argv[0], '\\')+1);
        else if(strrchr(argv[0], '/')) strcpy (name, strrchr(argv[0], '/')+1);
        else strcpy (name, *argv[0]?argv[0]:"Tachyon");
    }
    for (i = 1; i < argc; i++) {
        strcat (name, " ");
        strcat (name, argv[i]);
    }
#ifdef _DEBUG
    strcat (name, " (DEBUG BUILD)");
#endif
    return name;
}

int useoptions(argoptions * opt, SceneHandle scene) {
  if (opt->useoutfilename == 1) {
    rt_outputfile(scene, opt->outfilename);
  }

  if (opt->verbosemode == 1) {
    rt_verbose(scene, 1);
  }

  if (opt->antialiasing != -1) {
    /* need new api code for this */
  } 

  if (opt->displaymode != -1) {
    rt_displaymode(scene, opt->displaymode);
  }

  if (opt->boundmode != -1) {
    rt_boundmode(scene, opt->boundmode);
  }

  if (opt->boundthresh != -1) {
    rt_boundthresh(scene, opt->boundthresh);
  }

  return 0;
}    

argoptions ParseCommandLine(int argc, const char *argv[]) {
    argoptions opt;

    initoptions(&opt);

    bool nobounding = false;
    bool nodisp = false;

    string filename;

    utility::parse_cli_arguments(argc,argv,
        utility::cli_argument_pack()
        .positional_arg(filename,"dataset", "Model file")
        .positional_arg(opt.boundthresh,"boundthresh","bounding threshold value")
        .arg(nodisp,"no-display-updating","disable run-time display updating")
        .arg(nobounding,"no-bounding","disable bounding technique")
        .arg(silent_mode,"silent","no output except elapsed time")
    );

    strcpy(opt.filename, filename.c_str());

    opt.displaymode = nodisp ? RT_DISPLAY_DISABLED : RT_DISPLAY_ENABLED;
    opt.boundmode = nobounding ? RT_BOUNDING_DISABLED : RT_BOUNDING_ENABLED;

    return opt;
}

int CreateScene(argoptions &opt) {
    char *filename;

    global_scene = rt_newscene();
    rt_initialize();

    /* process command line overrides */
    useoptions(&opt, global_scene);

#ifdef DEFAULT_MODELFILE
#if  _WIN32||_WIN64
#define _GLUE_FILENAME(x) "..\\dat\\" #x
#else
#define _GLUE_FILENAME(x) #x
#endif
#define GLUE_FILENAME(x) _GLUE_FILENAME(x)
    if(opt.foundfilename == -1)
        filename = GLUE_FILENAME(DEFAULT_MODELFILE);
    else
#endif//DEFAULT_MODELFILE
        filename = opt.filename;

    if ( readmodel(filename, global_scene) != 0 ) {
        fprintf(stderr, "Parser returned a non-zero error code reading %s\n", filename);
        fprintf(stderr, "Aborting Render...\n");
        rt_finalize();
        return -1;
    }

    // need these early for create_graphics_window() so grab these here...
    scenedef *scene = (scenedef *) global_scene;
    global_xsize = scene->hres;
    global_ysize = scene->vres;
    global_xwinsize = global_xsize;
    global_ywinsize = global_ysize;  // add some here to leave extra blank space on bottom for status etc.

    return 0;
}

int main (int argc, char *argv[]) {
    try {
        timer mainStartTime = gettimer();

        global_window_title = window_title_string (argc, (const char**)argv);

        argoptions opt = ParseCommandLine(argc, (const char**)argv);

        if ( CreateScene(opt) != 0 )
            return -1;

        tachyon_video tachyon;
        tachyon.threaded = true;
        tachyon.init_console();

        tachyon.title = global_window_title;
        // always using window even if(!global_usegraphics)
        global_usegraphics = 
            tachyon.init_window(global_xwinsize, global_ywinsize);
        if(!tachyon.running)
            return -1;

        video = &tachyon;
        tachyon.main_loop();

        utility::report_elapsed_time(timertime(mainStartTime, gettimer()));
        return 0;
    } catch ( std::exception& e ) {
        std::cerr<<"error occurred. error text is :\"" <<e.what()<<"\"\n";
        return 1;
    }
}
#endif

