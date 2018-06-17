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
 * box.cpp - This file contains the functions for dealing with boxes.
 */
 
#include "machine.h"
#include "types.h"
#include "macros.h"
#include "box.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

int box_bbox(void * obj, vector * min, vector * max) {
  box * b = (box *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}

static object_methods box_methods = {
  (void (*)(void *, void *))(box_intersect),
  (void (*)(void *, void *, void *, void *))(box_normal),
  box_bbox, 
  free 
};

box * newbox(void * tex, vector min, vector max) {
  box * b;
  
  b=(box *) rt_getmem(sizeof(box));
  memset(b, 0, sizeof(box));
  b->methods = &box_methods;
  b->tex = (texture *)tex;
  b->min = min; 
  b->max = max;

  return b;
}

void box_intersect(box * bx, ray * ry) {
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;

  tnear= -FHUGE;
  tfar= FHUGE;

  if (ry->d.x == 0.0) {
    if ((ry->o.x < bx->min.x) || (ry->o.x > bx->max.x)) return;
  }
  else {
    tx1 = (bx->min.x - ry->o.x) / ry->d.x;
    tx2 = (bx->max.x - ry->o.x) / ry->d.x;
    if (tx1 > tx2) { a=tx1; tx1=tx2; tx2=a; } 
    if (tx1 > tnear) tnear=tx1;   
    if (tx2 < tfar)   tfar=tx2;   
  } 
  if (tnear > tfar) return; 
  if (tfar < 0.0) return;
  
  if (ry->d.y == 0.0) { 
    if ((ry->o.y < bx->min.y) || (ry->o.y > bx->max.y)) return;
  }
  else {
    ty1 = (bx->min.y - ry->o.y) / ry->d.y;
    ty2 = (bx->max.y - ry->o.y) / ry->d.y;
    if (ty1 > ty2) { a=ty1; ty1=ty2; ty2=a; } 
    if (ty1 > tnear) tnear=ty1;   
    if (ty2 < tfar)   tfar=ty2;   
  }
  if (tnear > tfar) return; 
  if (tfar < 0.0) return;
 
  if (ry->d.z == 0.0) { 
    if ((ry->o.z < bx->min.z) || (ry->o.z > bx->max.z)) return;
  }
  else {
    tz1 = (bx->min.z - ry->o.z) / ry->d.z;
    tz2 = (bx->max.z - ry->o.z) / ry->d.z;
    if (tz1 > tz2) { a=tz1; tz1=tz2; tz2=a; } 
    if (tz1 > tnear) tnear=tz1;   
    if (tz2 < tfar)   tfar=tz2;   
  }
  if (tnear > tfar) return; 
  if (tfar < 0.0) return;

  add_intersection(tnear, (object *) bx, ry);
  add_intersection(tfar, (object *) bx, ry);
}

void box_normal(box * bx, vector  * pnt, ray * incident, vector * N) {
  vector a, b, c; 
  flt t;
 
  c.x=(bx->max.x + bx->min.x) / 2.0;
  c.y=(bx->max.y + bx->min.y) / 2.0;
  c.z=(bx->max.z + bx->min.z) / 2.0;
 
  VSub((vector *) pnt, &c, N);
  b=(*N);

  a.x=fabs(N->x);
  a.y=fabs(N->y);
  a.z=fabs(N->z);
 
  N->x=0.0;  N->y=0.0;  N->z=0.0;

  t=MYMAX(a.x, MYMAX(a.y, a.z));  

  if (t==a.x) N->x=b.x;  

  if (t==a.y) N->y=b.y; 

  if (t==a.z) N->z=b.z;

  VNorm(N);
}

