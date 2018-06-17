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
 * quadric.cpp - This file contains the functions for dealing with quadrics.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "quadric.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

int quadric_bbox(void * obj, vector * min, vector * max) {
  return 0;
}

static object_methods quadric_methods = {
  (void (*)(void *, void *))(quadric_intersect),
  (void (*)(void *, void *, void *, void *))(quadric_normal),
  quadric_bbox, 
  free 
};
 
quadric * newquadric() {
  quadric * q;
 
  q=(quadric *) rt_getmem(sizeof(quadric));
  memset(q, 0, sizeof(quadric));
  q->ctr.x=0.0;
  q->ctr.y=0.0;
  q->ctr.z=0.0;
  q->methods = &quadric_methods;
 
  return q;
}

void quadric_intersect(quadric * q, ray * ry) {
  flt Aq, Bq, Cq;
  flt t1, t2;
  flt disc;
  vector rd;
  vector ro;
 
  rd=ry->d;
  VNorm(&rd);

  ro.x =  ry->o.x - q->ctr.x;
  ro.y =  ry->o.y - q->ctr.y;
  ro.z =  ry->o.z - q->ctr.z;


  Aq = (q->mat.a*(rd.x * rd.x)) +
        (2.0 * q->mat.b * rd.x * rd.y) +
        (2.0 * q->mat.c * rd.x * rd.z) +
        (q->mat.e * (rd.y * rd.y)) +
        (2.0 * q->mat.f * rd.y * rd.z) +
        (q->mat.h * (rd.z * rd.z));

  Bq = 2.0 * (
        (q->mat.a * ro.x * rd.x) +
        (q->mat.b * ((ro.x * rd.y) + (rd.x * ro.y))) +
        (q->mat.c * ((ro.x * rd.z) + (rd.x * ro.z))) +
        (q->mat.d * rd.x) +
        (q->mat.e * ro.y * rd.y) +
        (q->mat.f * ((ro.y * rd.z) + (rd.y * ro.z))) +
        (q->mat.g * rd.y) +
        (q->mat.h * ro.z * rd.z) +
        (q->mat.i * rd.z)
        );

  Cq = (q->mat.a * (ro.x * ro.x)) +
        (2.0 * q->mat.b * ro.x * ro.y) +
        (2.0 * q->mat.c * ro.x * ro.z) +
        (2.0 * q->mat.d * ro.x) +
        (q->mat.e * (ro.y * ro.y)) +
        (2.0 * q->mat.f * ro.y * ro.z) +
        (2.0 * q->mat.g * ro.y) +
        (q->mat.h * (ro.z * ro.z)) +
        (2.0 * q->mat.i * ro.z) +
        q->mat.j;

  if (Aq == 0.0) {
          t1 = - Cq / Bq;
          add_intersection(t1, (object *) q, ry);
          }
  else {
    disc=(Bq*Bq - 4.0 * Aq * Cq);
    if (disc > 0.0) {
          disc=sqrt(disc);
          t1 = (-Bq + disc) / (2.0 * Aq);
          t2 = (-Bq - disc) / (2.0 * Aq);
          add_intersection(t1, (object *) q, ry);
          add_intersection(t2, (object *) q, ry); 
          }
  }
}

void quadric_normal(quadric * q, vector * pnt, ray * incident, vector * N) {

  N->x = (q->mat.a*(pnt->x - q->ctr.x) + 
	  q->mat.b*(pnt->y - q->ctr.y) + 
	  q->mat.c*(pnt->z - q->ctr.z) + q->mat.d);

  N->y = (q->mat.b*(pnt->x - q->ctr.x) + 
	  q->mat.e*(pnt->y - q->ctr.y) + 
	  q->mat.f*(pnt->z - q->ctr.z) + q->mat.g);

  N->z = (q->mat.c*(pnt->x - q->ctr.x) + 
	  q->mat.f*(pnt->y - q->ctr.y) + 
	  q->mat.h*(pnt->z - q->ctr.z) + q->mat.i);

  VNorm(N);

  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  } 
}
 

