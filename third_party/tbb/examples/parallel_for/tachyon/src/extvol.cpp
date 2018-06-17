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
 * extvol.cpp - Volume rendering helper routines etc.
 */

#include<stdio.h>

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "util.h"
#include "box.h"
#include "extvol.h"
#include "trace.h"
#include "sphere.h"
#include "light.h"
#include "shade.h"
#include "global.h"


int extvol_bbox(void * obj, vector * min, vector * max) {
  box * b = (box *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}

static object_methods extvol_methods = {
  (void (*)(void *, void *))(box_intersect),
  (void (*)(void *, void *, void *, void *))(box_normal),
  extvol_bbox, 
  free 
};

extvol * newextvol(void * voidtex, vector min, vector max, 
                   int samples, flt (* evaluator)(flt, flt, flt)) { 
  extvol * xvol;
  texture * tex;
  
  tex = (texture *) voidtex;

  xvol = (extvol *) rt_getmem(sizeof(extvol));
  memset(xvol, 0, sizeof(extvol));

  xvol->methods = &extvol_methods;

  xvol->min=min;
  xvol->max=max;
  xvol->evaluator = evaluator;
  xvol->ambient = tex->ambient;
  xvol->diffuse = tex->diffuse;
  xvol->opacity = tex->opacity;  
  xvol->samples = samples;

  xvol->tex = (texture *)rt_getmem(sizeof(texture));
  memset(xvol->tex, 0, sizeof(texture));

  xvol->tex->ctr.x = 0.0;
  xvol->tex->ctr.y = 0.0;
  xvol->tex->ctr.z = 0.0;
  xvol->tex->rot = xvol->tex->ctr;
  xvol->tex->scale = xvol->tex->ctr;
  xvol->tex->uaxs = xvol->tex->ctr;
  xvol->tex->vaxs = xvol->tex->ctr;
  xvol->tex->islight = 0;
  xvol->tex->shadowcast = 0;

  xvol->tex->col=tex->col;
  xvol->tex->ambient=1.0;
  xvol->tex->diffuse=0.0;
  xvol->tex->specular=0.0;
  xvol->tex->opacity=1.0;
  xvol->tex->img=NULL;
  xvol->tex->texfunc=(color(*)(void *, void *, void *))(ext_volume_texture);
  xvol->tex->obj = (void *) xvol; /* XXX hack! */

  return xvol;
}

color ExtVoxelColor(flt scalar) {
  color col;

  if (scalar > 1.0) 
    scalar = 1.0;

  if (scalar < 0.0)
    scalar = 0.0;

  if (scalar < 0.5) {
    col.g = 0.0;
  }
  else {
    col.g = (scalar - 0.5) * 2.0;
  }

  col.r = scalar;
  col.b = 1.0 - (scalar / 2.0);

  return col;
} 

color ext_volume_texture(vector * hit, texture * tex, ray * ry) {
  color col, col2;
  box * bx;
  extvol * xvol;
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;
  flt t, tdist, dt, ddt, sum, tt; 
  vector pnt, bln;
  flt scalar, transval; 
  int i;
  point_light * li;
  color diffint; 
  vector N, L;
  flt inten;

  col.r = 0.0;
  col.g = 0.0;
  col.b = 0.0;

    bx = (box *) tex->obj;
  xvol = (extvol *) tex->obj;
 
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
 
  tdist = xvol->samples;

  tt = (xvol->opacity / tdist); 

  bln.x=fabs(bx->min.x - bx->max.x);
  bln.y=fabs(bx->min.y - bx->max.y);
  bln.z=fabs(bx->min.z - bx->max.z);
  
     dt = 1.0 / tdist; 
    sum = 0.0;

/* Accumulate color as the ray passes through the voxels */
  for (t=tnear; t<=tfar; t+=dt) {
    if (sum < 1.0) {
      pnt.x=((ry->o.x + (ry->d.x * t)) - bx->min.x) / bln.x;
      pnt.y=((ry->o.y + (ry->d.y * t)) - bx->min.y) / bln.y;
      pnt.z=((ry->o.z + (ry->d.z * t)) - bx->min.z) / bln.z;

      /* call external evaluator assume 0.0 -> 1.0 range.. */ 
      scalar = xvol->evaluator(pnt.x, pnt.y, pnt.z);  

      transval = tt * scalar; 
      sum += transval; 

      col2 = ExtVoxelColor(scalar);

      col.r += transval * col2.r * xvol->ambient;
      col.g += transval * col2.g * xvol->ambient;
      col.b += transval * col2.b * xvol->ambient;

      ddt = dt;

      /* Add in diffuse shaded light sources (no shadows) */
      if (xvol->diffuse > 0.0) {
  
        /* Calculate the Volume gradient at the voxel */
        N.x = (xvol->evaluator(pnt.x - ddt, pnt.y, pnt.z)  -  
              xvol->evaluator(pnt.x + ddt, pnt.y, pnt.z)) *  8.0 * tt; 
  
        N.y = (xvol->evaluator(pnt.x, pnt.y - ddt, pnt.z)  -  
              xvol->evaluator(pnt.x, pnt.y + ddt, pnt.z)) *  8.0 * tt; 
  
        N.z = (xvol->evaluator(pnt.x, pnt.y, pnt.z - ddt)  -  
              xvol->evaluator(pnt.x, pnt.y, pnt.z + ddt)) *  8.0 * tt; 
 
        /* only light surfaces with enough of a normal.. */
        if ((N.x*N.x + N.y*N.y + N.z*N.z) > 0.0) { 
          diffint.r = 0.0; 
          diffint.g = 0.0; 
          diffint.b = 0.0; 
    
          /* add the contribution of each of the lights.. */
          for (i=0; i<numlights; i++) {
            li=lightlist[i];
            VSUB(li->ctr, (*hit), L)
            VNorm(&L);
            VDOT(inten, N, L)
    
            /* only add light if its from the front of the surface */
            /* could add back-lighting if we wanted to later.. */
            if (inten > 0.0) {
              diffint.r += inten*li->tex->col.r;
              diffint.g += inten*li->tex->col.g;
              diffint.b += inten*li->tex->col.b;
            }
          }   
          col.r += col2.r * diffint.r * xvol->diffuse;
          col.g += col2.g * diffint.g * xvol->diffuse;
          col.b += col2.b * diffint.b * xvol->diffuse;
        }
      }
    }   
    else { 
      sum=1.0;
    }  
  }

  /* Add in transmitted ray from outside environment */
  if (sum < 1.0) {      /* spawn transmission rays / refraction */
    color transcol;

    transcol = shade_transmission(ry, hit, 1.0 - sum);

    col.r += transcol.r; /* add the transmitted ray  */
    col.g += transcol.g; /* to the diffuse and       */
    col.b += transcol.b; /* transmission total..     */
  }

  return col;
}



