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
 * light.cpp - This file contains declarations and defines for light sources.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

#define LIGHT_PRIVATE
#include "light.h"

static object_methods light_methods = {
  (void (*)(void *, void *))(light_intersect),
  (void (*)(void *, void *, void *, void *))(light_normal),
  light_bbox, 
  free 
};

point_light * newlight(void * tex, vector ctr, flt rad) {
  point_light * l;
  
  l=(point_light *) rt_getmem(sizeof(point_light));
  memset(l, 0, sizeof(point_light));
  l->methods = &light_methods;

  l->tex=(texture *)tex;
  l->ctr=ctr;
  l->rad=rad;

  return l;
}

static int light_bbox(void * obj, vector * min, vector * max) {
  return 0; /* lights are unbounded currently */
}

static void light_intersect(point_light * l, ray * ry) {
  flt b, disc, t1, t2, temp;
  vector V;

  /* Lights do not cast shadows.. */
  if (ry->flags & RT_RAY_SHADOW)
    return;

  VSUB(l->ctr, ry->o, V);
  VDOT(b, V, ry->d); 
  VDOT(temp, V, V);  

  disc=b*b + l->rad*l->rad - temp;

  if (disc<=0.0) return;
  disc=sqrt(disc);

  t2=b+disc;
  if (t2 <= SPEPSILON) 
    return;
  add_intersection(t2, (object *) l, ry);  

  t1=b-disc;
  if (t1 > SPEPSILON) 
    add_intersection(t1, (object *) l, ry);  
}

static void light_normal(point_light * l, vector * pnt, ray * incident, vector * N) {
  VSub((vector *) pnt, &(l->ctr), N);

  VNorm(N);

  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  } 
}


