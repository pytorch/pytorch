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
 * intersect.cpp - This file contains code for CSG and intersection routines.
 */

#include "machine.h"
#include "types.h"
#include "intersect.h"
#include "light.h"
#include "util.h"
#include "global.h"

unsigned int new_objectid(void) {
  return numobjects++; /* global used to generate unique object ID's */
}

unsigned int max_objectid(void) {
  return numobjects;
}

void add_object(object * obj) {
  object * objtemp;

  if (obj == NULL)
    return;

  obj->id = new_objectid();

  objtemp = rootobj;
  rootobj = obj;
  obj->nextobj = objtemp;
}

void free_objects(object * start) {
  object * cur;
  object * cur2;

  cur=start; 
  while (cur->nextobj != NULL) { 
    cur2=(object *)cur->nextobj;
    cur->methods->free(cur);
    cur=cur2;
  }
  free(cur);

}

void reset_object(void) {
  if (rootobj != NULL)
    free_objects(rootobj);

  rootobj = NULL;
  numobjects = 0; /* set number of objects back to 0 */
}

void intersect_objects(ray * intray) {
  object * cur;
  object temp;

  temp.nextobj = rootobj; /* setup the initial object pointers.. */
  cur = &temp;            /* ready, set                          */

  while ((cur=(object *)cur->nextobj) != NULL)          
    cur->methods->intersect(cur, intray); 
}

void reset_intersection(intersectstruct * intstruct) {
  intstruct->num = 0;
  intstruct->list[0].t = FHUGE;
  intstruct->list[0].obj = NULL;
  intstruct->list[1].t = FHUGE;
  intstruct->list[1].obj = NULL;
}

void add_intersection(flt t, object * obj, ray * ry) {
  intersectstruct * intstruct = ry->intstruct;

  if (t > EPSILON) {

    /* if we hit something before maxdist update maxdist */
    if (t < ry->maxdist) {
      ry->maxdist = t;

      /* if we hit *anything* before maxdist, and we're firing a */
      /* shadow ray, then we are finished ray tracing the shadow */
      if (ry->flags & RT_RAY_SHADOW)
        ry->flags |= RT_RAY_FINISHED;
    }

    intstruct->num++;
    intstruct->list[intstruct->num].obj = obj;
    intstruct->list[intstruct->num].t = t;
  }
}


int closest_intersection(flt * t, object ** obj, intersectstruct * intstruct) {
  int i;
  *t=FHUGE;

  for (i=1; i<=intstruct->num; i++) {
    if (intstruct->list[i].t < *t) {
        *t=intstruct->list[i].t;
      *obj=intstruct->list[i].obj;
    }
  } 

  return intstruct->num;
}

int shadow_intersection(intersectstruct * intstruct, flt maxdist) {
  int i;
  
  if (intstruct->num > 0) {
    for (i=1; i<=intstruct->num; i++) {
      if ((intstruct->list[i].t < maxdist) && 
          (intstruct->list[i].obj->tex->shadowcast == 1)) {
        return 1;
      }
    }
  }
  
  return 0;
}





