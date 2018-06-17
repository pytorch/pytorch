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
 * render.cpp - This file contains the main program and driver for the raytracer.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "tgafile.h"
#include "trace.h"
#include "render.h"
#include "util.h"
#include "light.h"
#include "global.h"
#include "ui.h"
#include "tachyon_video.h"
#include "objbound.h"
#include "grid.h"

/* how many pieces to divide each scanline into */
#define NUMHORZDIV 1  

void renderscene(scenedef scene) {
  //char msgtxt[2048];
  //void * outfile;
  /* Grid based accerlation scheme */
  if (scene.boundmode == RT_BOUNDING_ENABLED) 
    engrid_scene(&rootobj); /* grid */
  /* Not used now
  if (scene.verbosemode) { 
    sprintf(msgtxt, "Opening %s for output.", scene.outfilename); 
    rt_ui_message(MSG_0, msgtxt);
  }

  createtgafile(scene.outfilename,  
                  (unsigned short) scene.hres, 
                  (unsigned short) scene.vres);
  outfile = opentgafile(scene.outfilename);
  */

  trace_region (scene, 0/*outfile*/, 0, 0, scene.hres, scene.vres);
  //fclose((FILE *)outfile);
} /* end of renderscene() */
