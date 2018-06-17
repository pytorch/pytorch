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

/* 
 * trace.cpp - This file contains the functions for firing primary rays
 *           and handling subsequent calculations
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

color trace(ray * primary) {
  if (primary->depth > 0) {
    VNorm(&primary->d);
    reset_intersection(primary->intstruct);
    intersect_objects(primary);
    return shader(primary);
  }

  /* if ray is truncated, return the background as its color */
  return primary->scene->background;
}

void * thread_io(void * parms) {
  thr_io_parms p;

  p= *((thr_io_parms *) parms);
  writetgaregion(p.tga, p.iwidth, p.iheight, p.startx, p.starty, 
		p.stopx, p.stopy, p.buffer);
  free(p.buffer); /* free the buffer once we are done with it.. */
  free(parms);

  return(NULL);
}

void trace_shm(scenedef scene, /*char * buffer,  */ int startx, int stopx, int starty, int stopy) {

  thr_parms * parms;

  parms = (thr_parms *) rt_getmem(sizeof(thr_parms));  

  parms->tid=0;
  parms->nthr=1;
  parms->scene=scene;
  parms->startx=startx;
  parms->stopx=stopx;
  parms->starty=starty;
  parms->stopy=stopy;

  thread_trace(parms);

  rt_freemem(parms);
}

void trace_region(scenedef scene, void * tga, int startx, int starty, int stopx, int stopy) {

  if (scene.verbosemode) {
    char msgtxt[2048];
    sprintf(msgtxt, "Node %3d tracing region  %4d, %4d  --->  %4d, %4d \n", 0, startx,starty,stopx,stopy);
    rt_ui_message(MSG_0, msgtxt);
  }

  trace_shm(scene, /*buffer,*/ startx, stopx, starty, stopy);
/* not used now
  writetgaregion(tga, scene.hres, scene.vres, 
                 startx, starty, stopx, stopy, global_buffer);

  if (scene.rawimage != NULL) {
    int x, y;
    int totalx = stopx - startx + 1;
    for (y=starty; y<=stopy; y++) {
      for (x=0; x<scene.hres; x++) {
        scene.rawimage[(scene.vres-y)*scene.hres*3 + x*3] = global_buffer[(y-starty)*totalx*3 + x*3 + 2];
        scene.rawimage[(scene.vres-y)*scene.hres*3 + x*3 +1] = global_buffer[(y-starty)*totalx*3 + x*3 + 1];
        scene.rawimage[(scene.vres-y)*scene.hres*3 + x*3 +2] = global_buffer[(y-starty)*totalx*3 + x*3];
      }
    }
  }
*/
}
