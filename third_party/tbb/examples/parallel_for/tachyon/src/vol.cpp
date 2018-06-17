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
 * vol.cpp - Volume rendering helper routines etc.
 */

#include <stdio.h>
#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "util.h"
#include "vol.h"
#include "box.h"
#include "trace.h"
#include "ui.h"
#include "light.h"
#include "shade.h"

int scalarvol_bbox(void * obj, vector * min, vector * max) {
  box * b = (box *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}

void * newscalarvol(void * intex, vector min, vector max,
                    int xs, int ys, int zs, char * fname, scalarvol * invol) {
  box * bx;
  texture * tx, * tex;
  scalarvol * vol;

  tex=(texture *)intex;
  tex->shadowcast = 0; /* doesn't cast a shadow */

  tx=(texture *)rt_getmem(sizeof(texture));

  /* is the volume data already loaded? */
  if (invol==NULL) {
    vol=(scalarvol *)rt_getmem(sizeof(scalarvol));
    vol->loaded=0;
    vol->data=NULL;
  }
  else
    vol=invol;

  vol->opacity=tex->opacity;
  vol->xres=xs;
  vol->yres=ys;
  vol->zres=zs;
  strcpy(vol->name, fname);

  tx->ctr.x = 0.0;
  tx->ctr.y = 0.0;
  tx->ctr.z = 0.0;
  tx->rot   = tx->ctr;
  tx->scale = tx->ctr;
  tx->uaxs  = tx->ctr;
  tx->vaxs  = tx->ctr;

  tx->islight = 0;
  tx->shadowcast = 0; /* doesn't cast a shadow */

  tx->col = tex->col;
  tx->ambient  = 1.0;
  tx->diffuse  = 0.0;
  tx->specular = 0.0;
  tx->opacity  = 1.0;
  tx->img = vol;
  tx->texfunc = (color(*)(void *, void *, void *))(scalar_volume_texture);

  bx=newbox(tx, min, max);
  tx->obj = (void *) bx; /* XXX hack! */

  return (void *) bx;
}


color VoxelColor(flt scalar) {
  color col;

  if (scalar > 1.0)
    scalar = 1.0;

  if (scalar < 0.0)
    scalar = 0.0;

  if (scalar < 0.25) {
    col.r = scalar * 4.0;
    col.g = 0.0;
    col.b = 0.0;
  }
  else {
    if (scalar < 0.75) {
      col.r = 1.0;
      col.g = (scalar - 0.25) * 2.0;
      col.b = 0.0;
    }
    else {
      col.r = 1.0;
      col.g = 1.0;
      col.b = (scalar - 0.75) * 4.0;
    }
  }

  return col;
}

color scalar_volume_texture(vector * hit, texture * tex, ray * ry) {
  color col, col2;
  box * bx;
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;
  flt t, tdist, dt, sum, tt;
  vector pnt, bln;
  scalarvol * vol;
  flt scalar, transval;
  int x, y, z;
  unsigned char * ptr;

  bx=(box *) tex->obj;
  vol=(scalarvol *)bx->tex->img;

  col.r=0.0;
  col.g=0.0;
  col.b=0.0;

  tnear= -FHUGE;
  tfar= FHUGE;

  if (ry->d.x == 0.0) {
    if ((ry->o.x < bx->min.x) || (ry->o.x > bx->max.x)) return col;
  }
  else {
    tx1 = (bx->min.x - ry->o.x) / ry->d.x;
    tx2 = (bx->max.x - ry->o.x) / ry->d.x;
    if (tx1 > tx2) { a=tx1; tx1=tx2; tx2=a; }
    if (tx1 > tnear) tnear=tx1;
    if (tx2 < tfar)   tfar=tx2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;

 if (ry->d.y == 0.0) {
    if ((ry->o.y < bx->min.y) || (ry->o.y > bx->max.y)) return col;
  }
  else {
    ty1 = (bx->min.y - ry->o.y) / ry->d.y;
    ty2 = (bx->max.y - ry->o.y) / ry->d.y;
    if (ty1 > ty2) { a=ty1; ty1=ty2; ty2=a; }
    if (ty1 > tnear) tnear=ty1;
    if (ty2 < tfar)   tfar=ty2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;

  if (ry->d.z == 0.0) {
    if ((ry->o.z < bx->min.z) || (ry->o.z > bx->max.z)) return col;
  }
  else {
    tz1 = (bx->min.z - ry->o.z) / ry->d.z;
    tz2 = (bx->max.z - ry->o.z) / ry->d.z;
    if (tz1 > tz2) { a=tz1; tz1=tz2; tz2=a; }
    if (tz1 > tnear) tnear=tz1;
    if (tz2 < tfar)   tfar=tz2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;

  if (tnear < 0.0) tnear=0.0;

  tdist=sqrt((flt) (vol->xres*vol->xres + vol->yres*vol->yres + vol->zres*vol->zres));
  tt = (vol->opacity / tdist);

  bln.x=fabs(bx->min.x - bx->max.x);
  bln.y=fabs(bx->min.y - bx->max.y);
  bln.z=fabs(bx->min.z - bx->max.z);

  dt=sqrt(bln.x*bln.x + bln.y*bln.y + bln.z*bln.z) / tdist;
  sum=0.0;

  /* move the volume residency check out of loop.. */
  if (!vol->loaded) {
    LoadVol(vol);
    vol->loaded=1;
  }

  for (t=tnear; t<=tfar; t+=dt) {
    pnt.x=((ry->o.x + (ry->d.x * t)) - bx->min.x) / bln.x;
    pnt.y=((ry->o.y + (ry->d.y * t)) - bx->min.y) / bln.y;
    pnt.z=((ry->o.z + (ry->d.z * t)) - bx->min.z) / bln.z;

    x=(int) ((vol->xres - 1.5) * pnt.x + 0.5);
    y=(int) ((vol->yres - 1.5) * pnt.y + 0.5);
    z=(int) ((vol->zres - 1.5) * pnt.z + 0.5);

    ptr = vol->data + ((vol->xres * vol->yres * z) + (vol->xres * y) + x);

    scalar = (flt) ((flt) 1.0 * ((int) ptr[0])) / 255.0;

    sum += tt * scalar;

    transval = tt * scalar;

    col2 = VoxelColor(scalar);

    if (sum < 1.0) {
      col.r += transval * col2.r;
      col.g += transval * col2.g;
      col.b += transval * col2.b;
      if (sum < 0.0) sum=0.0;
    }
    else {
      sum=1.0;
    }
  }

  if (sum < 1.0) {      /* spawn transmission rays / refraction */
    color transcol;

    transcol = shade_transmission(ry, hit, 1.0 - sum);

    col.r += transcol.r; /* add the transmitted ray  */
    col.g += transcol.g; /* to the diffuse and       */
    col.b += transcol.b; /* transmission total..     */
  }

  return col;
}

void LoadVol(scalarvol * vol) {
  FILE * dfile;
  size_t status;
  char msgtxt[2048];

  dfile=fopen(vol->name, "r");
  if (dfile==NULL) {
    char msgtxt[2048];
    sprintf(msgtxt, "Vol: can't open %s for input!!! Aborting\n",vol->name);
    rt_ui_message(MSG_ERR, msgtxt);
    rt_ui_message(MSG_ABORT, "Rendering Aborted.");
    exit(1);
  }

  sprintf(msgtxt, "loading %dx%dx%d volume set from %s",
      vol->xres, vol->yres, vol->zres, vol->name);
  rt_ui_message(MSG_0, msgtxt);

  vol->data = (unsigned char *)rt_getmem(vol->xres * vol->yres * vol->zres);

  status=fread(vol->data, 1, (vol->xres * vol->yres * vol->zres), dfile);
  fclose(dfile);
}
