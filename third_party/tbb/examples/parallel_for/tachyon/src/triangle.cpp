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
 * triangle.cpp - This file contains the functions for dealing with triangles.
 */
 
#include "machine.h"
#include "types.h"
#include "vector.h"
#include "macros.h"
#include "intersect.h"
#include "util.h"

#define TRIANGLE_PRIVATE
#include "triangle.h"

static object_methods tri_methods = {
  (void (*)(void *, void *))(tri_intersect),
  (void (*)(void *, void *, void *, void *))(tri_normal),
  tri_bbox, 
  free 
};

static object_methods stri_methods = {
  (void (*)(void *, void *))(tri_intersect),
  (void (*)(void *, void *, void *, void *))(stri_normal),
  tri_bbox, 
  free 
};

object * newtri(void * tex, vector v0, vector v1, vector v2) {
  tri * t;
  vector edge1, edge2, edge3;

  VSub(&v1, &v0, &edge1);
  VSub(&v2, &v0, &edge2);
  VSub(&v2, &v1, &edge3);

  /* check to see if this will be a degenerate triangle before creation */
  if ((VLength(&edge1) >= EPSILON) && 
      (VLength(&edge2) >= EPSILON) && 
      (VLength(&edge3) >= EPSILON)) {

    t=(tri *) rt_getmem(sizeof(tri));

    t->nextobj = NULL;
    t->methods = &tri_methods;

    t->tex = (texture *)tex;
    t->v0 = v0;
    t->edge1 = edge1;
    t->edge2 = edge2;
 
    return (object *) t;
  }
  
  return NULL; /* was a degenerate triangle */
}


object * newstri(void * tex, vector v0, vector v1, vector v2,
                           vector n0, vector n1, vector n2) {
  stri * t;
  vector edge1, edge2, edge3;

  VSub(&v1, &v0, &edge1);
  VSub(&v2, &v0, &edge2);
  VSub(&v2, &v1, &edge3);

  /* check to see if this will be a degenerate triangle before creation */
  if ((VLength(&edge1) >= EPSILON) && 
      (VLength(&edge2) >= EPSILON) &&
      (VLength(&edge3) >= EPSILON)) {

    t=(stri *) rt_getmem(sizeof(stri));

    t->nextobj = NULL;
    t->methods = &stri_methods;
 
    t->tex = (texture *)tex;
    t->v0 = v0;
    t->edge1 = edge1;
    t->edge2 = edge2;
    t->n0 = n0;
    t->n1 = n1;
    t->n2 = n2;

    return (object *) t;
  }

  return NULL; /* was a degenerate triangle */
}

#define CROSS(dest,v1,v2) \
          dest.x=v1.y*v2.z-v1.z*v2.y; \
          dest.y=v1.z*v2.x-v1.x*v2.z; \
          dest.z=v1.x*v2.y-v1.y*v2.x;

#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

#define SUB(dest,v1,v2) \
          dest.x=v1.x-v2.x; \
          dest.y=v1.y-v2.y; \
          dest.z=v1.z-v2.z;

static int tri_bbox(void * obj, vector * min, vector * max) {
  tri * t = (tri *) obj;
  vector v1, v2;

  VAdd(&t->v0, &t->edge1, &v1); 
  VAdd(&t->v0, &t->edge2, &v2); 

  min->x = MYMIN( t->v0.x , MYMIN( v1.x , v2.x ));
  min->y = MYMIN( t->v0.y , MYMIN( v1.y , v2.y ));
  min->z = MYMIN( t->v0.z , MYMIN( v1.z , v2.z ));

  max->x = MYMAX( t->v0.x , MYMAX( v1.x , v2.x ));
  max->y = MYMAX( t->v0.y , MYMAX( v1.y , v2.y ));
  max->z = MYMAX( t->v0.z , MYMAX( v1.z , v2.z ));

  return 1;
}

static void tri_intersect(tri * trn, ray * ry) {
  vector tvec, pvec, qvec;
  flt det, inv_det, t, u, v;

  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, ry->d, trn->edge2);

  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(trn->edge1, pvec);

   if (det > -EPSILON && det < EPSILON)
     return;

   inv_det = 1.0 / det;

   /* calculate distance from vert0 to ray origin */
   SUB(tvec, ry->o, trn->v0);

   /* calculate U parameter and test bounds */
   u = DOT(tvec, pvec) * inv_det;
   if (u < 0.0 || u > 1.0)
     return;

   /* prepare to test V parameter */
   CROSS(qvec, tvec, trn->edge1);

   /* calculate V parameter and test bounds */
   v = DOT(ry->d, qvec) * inv_det;
   if (v < 0.0 || u + v > 1.0)
     return;

   /* calculate t, ray intersects triangle */
   t = DOT(trn->edge2, qvec) * inv_det;

  add_intersection(t,(object *) trn, ry);
}


static void tri_normal(tri * trn, vector  * pnt, ray * incident, vector * N) {

  CROSS((*N), trn->edge1, trn->edge2);

  VNorm(N);

  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}

static void stri_normal(stri * trn, vector  * pnt, ray * incident, vector * N) {
  flt U, V, W, lensqr;
  vector P, tmp, norm;
  
  CROSS(norm, trn->edge1, trn->edge2);
  lensqr = DOT(norm, norm); 

  VSUB((*pnt), trn->v0, P);

  CROSS(tmp, P, trn->edge2);
  U = DOT(tmp, norm) / lensqr;   

  CROSS(tmp, trn->edge1, P);
  V = DOT(tmp, norm) / lensqr;   

  W = 1.0 - (U + V);

  N->x = W*trn->n0.x + U*trn->n1.x + V*trn->n2.x;
  N->y = W*trn->n0.y + U*trn->n1.y + V*trn->n2.y;
  N->z = W*trn->n0.z + U*trn->n1.z + V*trn->n2.z;

  VNorm(N);
}

