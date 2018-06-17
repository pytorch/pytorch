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

#include "types.h"
#include "api.h"       /* The ray tracing library API */
#include "ui.h"
#include "util.h"
#include "tachyon_video.h"

extern SceneHandle global_scene;
extern char *global_window_title;
extern bool global_usegraphics;

void tachyon_video::on_process() {
    char buf[8192];
    flt runtime;
    scenedef *scene = (scenedef *) global_scene;
    updating_mode = scene->displaymode == RT_DISPLAY_ENABLED;
    recycling = false;
    pausing = false;
    do {
        updating = updating_mode;
        timer start_timer = gettimer();
        rt_renderscene(global_scene);
        timer end_timer = gettimer();
        runtime = timertime(start_timer, end_timer);
        sprintf(buf, "%s: %.3f seconds", global_window_title, runtime);
        rt_ui_message(MSG_0, buf);
        title = buf; show_title(); // show time spent for rendering
        if(!updating) {
            updating = true;
            drawing_memory dm = get_drawing_memory();
            drawing_area drawing(0, 0, dm.sizex, dm.sizey);// invalidate whole screen
        }
        rt_finalize();
        title = global_window_title; show_title(); // reset title to default
    } while(recycling && running);
}

void tachyon_video::on_key(int key) {
    key &= 0xff;
    recycling = true;
    if(key == esc_key) running = false;
    else if(key == ' ') {
        if(!updating) {
            updating = true;
            drawing_memory dm = get_drawing_memory();
            drawing_area drawing(0, 0, dm.sizex, dm.sizey);// invalidate whole screen
        }
        updating = updating_mode = !updating_mode;
    }
    else if(key == 'p') {
        pausing = !pausing;
        if(pausing) {
            title = "Press ESC to exit or 'p' to continue after rendering completion";
            show_title();
        }
    }
}

void rt_finalize(void) {
    timer t0, t1;
    t0 = gettimer();
    if(global_usegraphics)
        do { rt_sleep(1); t1 = gettimer(); }
        while( (timertime(t0, t1) < 10 || video->pausing) && video->next_frame());
#ifdef _WINDOWS
    else rt_sleep(10000);
#endif
}
