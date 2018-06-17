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
 * api.cpp - This file contains all of the API calls that are defined for
 *         external driver code to use.  
 */

#include "machine.h"
#include "types.h"
#include "macros.h"

#include "box.h"
#include "cylinder.h"
#include "plane.h"
#include "quadric.h"
#include "ring.h"
#include "sphere.h"
#include "triangle.h"
#include "vol.h"
#include "extvol.h"

#include "texture.h"
#include "light.h"
#include "render.h"
#include "camera.h"
#include "vector.h"
#include "intersect.h"
#include "shade.h"
#include "util.h"
#include "imap.h"
#include "global.h"

#include "tachyon_video.h"

typedef void * SceneHandle;
#include "api.h"


vector rt_vector(apiflt x, apiflt y, apiflt z) {
  vector v;

  v.x = x;
  v.y = y;
  v.z = z;

  return v;
}

color rt_color(apiflt r, apiflt g, apiflt b) {
  color c;
  
  c.r = r;
  c.g = g;
  c.b = b;
  
  return c;
}

void rt_initialize() {
  rpcmsg msg;

  reset_object();
  reset_lights();
  InitTextures();

  if (!parinitted) {
    parinitted=1;

    msg.type=1; /* setup a ping message */
  }
}

void rt_renderscene(SceneHandle voidscene) {
  scenedef * scene = (scenedef *) voidscene;
  renderscene(*scene);
}

void rt_camerasetup(SceneHandle voidscene, apiflt zoom, apiflt aspectratio, 
	int antialiasing, int raydepth, 
	vector camcent, vector viewvec, vector upvec) {
  scenedef * scene = (scenedef *) voidscene;

  vector newupvec;
  vector newviewvec;
  vector newrightvec;
 
  VCross((vector *) &upvec, &viewvec, &newrightvec);
  VNorm(&newrightvec);

  VCross((vector *) &viewvec, &newrightvec, &newupvec);
  VNorm(&newupvec);

  newviewvec=viewvec;
  VNorm(&newviewvec);


  scene->camzoom=zoom; 
  scene->aspectratio=aspectratio;
  scene->antialiasing=antialiasing;
  scene->raydepth=raydepth; 
  scene->camcent=camcent;
  scene->camviewvec=newviewvec;
  scene->camrightvec=newrightvec;
  scene->camupvec=newupvec;
}

void rt_outputfile(SceneHandle voidscene, const char * outname) {
  scenedef * scene = (scenedef *) voidscene;
  strcpy((char *) &scene->outfilename, outname);
}

void rt_resolution(SceneHandle voidscene, int hres, int vres) {
  scenedef * scene = (scenedef *) voidscene;
  scene->hres=hres;
  scene->vres=vres;
}

void rt_verbose(SceneHandle voidscene, int v) {
  scenedef * scene = (scenedef *) voidscene;
  scene->verbosemode = v;
}

void rt_rawimage(SceneHandle voidscene, unsigned char *rawimage) {
  scenedef * scene = (scenedef *) voidscene;
  scene->rawimage = rawimage;
}

void rt_background(SceneHandle voidscene, color col) {
  scenedef * scene = (scenedef *) voidscene;
  scene->background.r = col.r;
  scene->background.g = col.g;
  scene->background.b = col.b;
}

void rt_boundmode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  scene->boundmode = mode;
}

void rt_boundthresh(SceneHandle voidscene, int threshold) {
  scenedef * scene = (scenedef *) voidscene;
 
  if (threshold > 1) {
    scene->boundthresh = threshold;
  }
  else {
    rtmesg("Ignoring out-of-range automatic bounding threshold.\n");
    rtmesg("Automatic bounding threshold reset to default.\n");
    scene->boundthresh = MAXOCTNODES;
  }
}

void rt_displaymode(SceneHandle voidscene, int mode) {
  scenedef * scene = (scenedef *) voidscene;
  scene->displaymode = mode;
}


void rt_scenesetup(SceneHandle voidscene, char * outname, int hres, int vres, int verbose) {
  rt_outputfile(voidscene, outname);
  rt_resolution(voidscene, hres, vres);
  rt_verbose(voidscene, verbose);
}

SceneHandle rt_newscene(void) {
  scenedef * scene;
  SceneHandle voidscene;

  scene = (scenedef *) malloc(sizeof(scenedef));
  memset(scene, 0, sizeof(scenedef));             /* clear all valuas to 0  */

  voidscene = (SceneHandle) scene;

  rt_outputfile(voidscene, "/dev/null");   /* default output file (.tga)   */
  rt_resolution(voidscene, 512, 512);             /* 512x512 resolution     */
  rt_verbose(voidscene, 0);                       /* verbose messages off   */
  rt_rawimage(voidscene, NULL);                   /* raw image output off   */
  rt_boundmode(voidscene, RT_BOUNDING_ENABLED);   /* spatial subdivision on */
  rt_boundthresh(voidscene, MAXOCTNODES);         /* default threshold      */
  rt_displaymode(voidscene, RT_DISPLAY_ENABLED);  /* video output on        */
  rt_camerasetup(voidscene, 1.0, 1.0, 0, 6,
                 rt_vector(0.0, 0.0, 0.0),
                 rt_vector(0.0, 0.0, 1.0),
                 rt_vector(0.0, 1.0, 0.0));
 
  return scene;
}

void rt_deletescene(SceneHandle scene) {
  if (scene != NULL)
    free(scene);
}

void apitextotex(apitexture * apitex, texture * tex) {
  switch(apitex->texturefunc) {
    case 0: 
      tex->texfunc=(color(*)(void *, void *, void *))(standard_texture);
      break;

    case 1: 
      tex->texfunc=(color(*)(void *, void *, void *))(checker_texture);
      break;

    case 2: 
      tex->texfunc=(color(*)(void *, void *, void *))(grit_texture);
      break;

    case 3: 
      tex->texfunc=(color(*)(void *, void *, void *))(marble_texture);
      break;

    case 4: 
      tex->texfunc=(color(*)(void *, void *, void *))(wood_texture);
      break;

    case 5: 
      tex->texfunc=(color(*)(void *, void *, void *))(gnoise_texture);
      break;
	
    case 6: 
      tex->texfunc=(color(*)(void *, void *, void *))(cyl_checker_texture);
      break;

    case 7: 
      tex->texfunc=(color(*)(void *, void *, void *))(image_sphere_texture);
      tex->img=AllocateImage((char *)apitex->imap);
      break;

    case 8: 
      tex->texfunc=(color(*)(void *, void *, void *))(image_cyl_texture);
      tex->img=AllocateImage((char *)apitex->imap);
      break;

    case 9: 
      tex->texfunc=(color(*)(void *, void *, void *))(image_plane_texture);
      tex->img=AllocateImage((char *)apitex->imap);
      break;

    default: 
      tex->texfunc=(color(*)(void *, void *, void *))(standard_texture);
      break;
  }

       tex->ctr = apitex->ctr;
       tex->rot = apitex->rot;
     tex->scale = apitex->scale;
      tex->uaxs = apitex->uaxs;
      tex->vaxs = apitex->vaxs;
   tex->ambient = apitex->ambient;
   tex->diffuse = apitex->diffuse;
  tex->specular = apitex->specular;
   tex->opacity = apitex->opacity;
       tex->col = apitex->col; 

  tex->islight = 0;
  tex->shadowcast = 1;
  tex->phong = 0.0;
  tex->phongexp = 0.0;
  tex->phongtype = 0;
}

void * rt_texture(apitexture * apitex) {
  texture * tex;

  tex=(texture *)rt_getmem(sizeof(texture));
  apitextotex(apitex, tex); 
  return(tex);
}

void rt_tex_color(void * voidtex, color col) {
  texture * tex = (texture *) voidtex;
  tex->col = col;
}

void rt_tex_phong(void * voidtex, apiflt phong, apiflt phongexp, int type) {
  texture * tex = (texture *) voidtex;
  tex->phong = phong;
  tex->phongexp = phongexp;
  tex->phongtype = type;
}

void rt_light(void * tex, vector ctr, apiflt rad) {
  point_light * li;

  li=newlight(tex, (vector) ctr, rad);

  li->tex->islight=1;
  li->tex->shadowcast=1;
  li->tex->diffuse=0.0;
  li->tex->specular=0.0;
  li->tex->opacity=1.0;

  add_light(li);
  add_object((object *)li);
}

void rt_scalarvol(void * tex, vector min, vector max,
	int xs, int ys, int zs, char * fname, void * invol) {
  add_object((object *) newscalarvol(tex, (vector)min, (vector)max, xs, ys, zs, fname, (scalarvol *) invol));
}

void rt_extvol(void * tex, vector min, vector max, int samples, flt (* evaluator)(flt, flt, flt)) {
  add_object((object *) newextvol(tex, (vector)min, (vector)max, samples, evaluator));
}

void rt_box(void * tex, vector min, vector max) {
  add_object((object *) newbox(tex, (vector)min, (vector)max));
} 

void rt_cylinder(void * tex, vector ctr, vector axis, apiflt rad) {
  add_object(newcylinder(tex, (vector)ctr, (vector)axis, rad));
}

void rt_fcylinder(void * tex, vector ctr, vector axis, apiflt rad) {
  add_object(newfcylinder(tex, (vector)ctr, (vector)axis, rad));
}

void rt_plane(void * tex, vector ctr, vector norm) {
  add_object(newplane(tex, (vector)ctr, (vector)norm));
} 

void rt_ring(void * tex, vector ctr, vector norm, apiflt a, apiflt b) {
  add_object(newring(tex, (vector)ctr, (vector)norm, a, b));
} 

void rt_sphere(void * tex, vector ctr, apiflt rad) {
  add_object(newsphere(tex, (vector)ctr, rad));
}

void rt_tri(void * tex, vector v0, vector v1, vector v2) {
  object * trn;

  trn = newtri(tex, (vector)v0, (vector)v1, (vector)v2);

  if (trn != NULL) { 
    add_object(trn);
  }
} 

void rt_stri(void * tex, vector v0, vector v1, vector v2, 
		vector n0, vector n1, vector n2) {
  object * trn;
 
  trn = newstri(tex, (vector)v0, (vector)v1, (vector)v2, (vector)n0, (vector)n1, (vector)n2);

  if (trn != NULL) { 
    add_object(trn);
  }
} 

void rt_quadsphere(void * tex, vector ctr, apiflt rad) {
  quadric * q;
  flt factor;
  q=(quadric *) newquadric();
  factor= 1.0 / (rad*rad);
  q->tex=(texture *)tex;
  q->ctr=ctr;
 
  q->mat.a=factor;
  q->mat.b=0.0;
  q->mat.c=0.0;
  q->mat.d=0.0;
  q->mat.e=factor;
  q->mat.f=0.0;
  q->mat.g=0.0;
  q->mat.h=factor;
  q->mat.i=0.0;
  q->mat.j=-1.0;
 
  add_object((object *)q);
}
