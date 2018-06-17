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
 * bndbox.cpp - This file contains the functions for dealing with bounding boxes.
 */
 
#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

#define BNDBOX_PRIVATE
#include "bndbox.h"

static object_methods bndbox_methods = {
  (void (*)(void *, void *))(bndbox_intersect),
  (void (*)(void *, void *, void *, void *))(NULL),
  bndbox_bbox, 
  free_bndbox 
};


bndbox * newbndbox(vector min, vector max) {
  bndbox * b;
  
  b=(bndbox *) rt_getmem(sizeof(bndbox));
  memset(b, 0, sizeof(bndbox));
  b->min=min;
  b->max=max;
  b->methods = &bndbox_methods;

  b->objlist=NULL;
  b->tex=NULL;
  b->nextobj=NULL;
  return b;
}


static int bndbox_bbox(void * obj, vector * min, vector * max) {
  bndbox * b = (bndbox *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}


static void free_bndbox(void * v) {
  bndbox * b = (bndbox *) v; 

  free_objects(b->objlist);  
 
  free(b);
}


static void bndbox_intersect(bndbox * bx, ray * ry) {
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;
  object * obj;
  ray newray; 

  /* eliminate bounded rays whose bounds do not intersect  */
  /* the bounds of the box..                               */
  if (ry->flags & RT_RAY_BOUNDED) {
    if ((ry->s.x > bx->max.x) && (ry->e.x > bx->max.x)) return;
    if ((ry->s.x < bx->min.x) && (ry->e.x < bx->min.x)) return;
  
    if ((ry->s.y > bx->max.y) && (ry->e.y > bx->max.y)) return;
    if ((ry->s.y < bx->min.y) && (ry->e.y < bx->min.y)) return;

    if ((ry->s.z > bx->max.z) && (ry->e.z > bx->max.z)) return;
    if ((ry->s.z < bx->min.z) && (ry->e.z < bx->min.z)) return;
  }

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


  /* intersect all of the enclosed objects */
  newray=*ry;
  newray.flags |= RT_RAY_BOUNDED;

  RAYPNT(newray.s , (*ry) , tnear); 
  RAYPNT(newray.e , (*ry) , (tfar + EPSILON)); 
 
  obj = bx->objlist;
  while (obj != NULL) {
    obj->methods->intersect(obj, &newray); 
    obj = (object *)obj->nextobj;
  }
}

