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
 * grid.cpp - spatial subdivision efficiency structures
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

#define GRID_PRIVATE
#include "grid.h"

#ifndef cbrt
#define     cbrt(x)     ((x) > 0.0 ? pow((double)(x), 1.0/3.0) : \
                          ((x) < 0.0 ? -pow((double)-(x), 1.0/3.0) : 0.0))

#define     qbrt(x)     ((x) > 0.0 ? pow((double)(x), 1.0/4.0) : \
                          ((x) < 0.0 ? -pow((double)-(x), 1.0/4.0) : 0.0))

#endif

static object_methods grid_methods = {
  (void (*)(void *, void *))(grid_intersect),
  (void (*)(void *, void *, void *, void *))(NULL),
  grid_bbox, 
  grid_free 
};

extern bool silent_mode;

object * newgrid(int xsize, int ysize, int zsize, vector min, vector max) {
  grid * g;

  g = (grid *) rt_getmem(sizeof(grid));
  memset(g, 0, sizeof(grid));  

  g->methods = &grid_methods;
  g->id = new_objectid();

  g->xsize = xsize;
  g->ysize = ysize;
  g->zsize = zsize;

  g->min = min;
  g->max = max;

  VSub(&g->max, &g->min, &g->voxsize);
  g->voxsize.x /= (flt) g->xsize; 
  g->voxsize.y /= (flt) g->ysize; 
  g->voxsize.z /= (flt) g->zsize; 

  g->cells = (objectlist **) rt_getmem(xsize*ysize*zsize*sizeof(objectlist *));
  memset(g->cells, 0, xsize*ysize*zsize * sizeof(objectlist *));

/* fprintf(stderr, "New grid, size: %8d %8d %8d\n", g->xsize, g->ysize, g->zsize); */

  return (object *) g;
}

static int grid_bbox(void * obj, vector * min, vector * max) {
  grid * g = (grid *) obj;
 
  *min = g->min;
  *max = g->max;

  return 1;
}

static void grid_free(void * v) {
  int i, numvoxels;
  grid * g = (grid *) v;
 
  /* loop through all voxels and free the object lists */
  numvoxels = g->xsize * g->ysize * g->zsize; 
  for (i=0; i<numvoxels; i++) {
    objectlist * lcur, * lnext;

    lcur = g->cells[i];
    while (lcur != NULL) {
      lnext = lcur->next;
      free(lcur);
    }
  }

  /* free the grid cells */ 
  free(g->cells);

  /* free all objects on the grid object list */
  free_objects(g->objects);   

  free(g);
}

static void globalbound(object ** rootlist, vector * gmin, vector * gmax) {
  vector min, max;
  object * cur;

  if (*rootlist == NULL)  /* don't bound non-existant objects */
    return;

  gmin->x =  FHUGE;   gmin->y =  FHUGE;   gmin->z =  FHUGE;
  gmax->x = -FHUGE;   gmax->y = -FHUGE;   gmax->z = -FHUGE;

  cur=*rootlist;
  while (cur != NULL)  {  /* Go! */
    min.x = -FHUGE; min.y = -FHUGE; min.z = -FHUGE;
    max.x =  FHUGE; max.y =  FHUGE; max.z =  FHUGE;

    if (cur->methods->bbox((void *) cur, &min, &max)) {
      gmin->x = MYMIN( gmin->x , min.x);
      gmin->y = MYMIN( gmin->y , min.y);
      gmin->z = MYMIN( gmin->z , min.z);

      gmax->x = MYMAX( gmax->x , max.x);
      gmax->y = MYMAX( gmax->y , max.y);
      gmax->z = MYMAX( gmax->z , max.z);
    }

    cur=(object *)cur->nextobj;
  }
}


static int cellbound(grid *g, gridindex *index, vector * cmin, vector * cmax) {
  vector min, max, cellmin, cellmax;
  objectlist * cur;
  int numinbounds = 0;

  cur = g->cells[index->z*g->xsize*g->ysize + index->y*g->xsize + index->x]; 

  if (cur == NULL)  /* don't bound non-existant objects */
    return 0;

  cellmin.x = voxel2x(g, index->x); 
  cellmin.y = voxel2y(g, index->y); 
  cellmin.z = voxel2z(g, index->z); 

  cellmax.x = cellmin.x + g->voxsize.x;
  cellmax.y = cellmin.y + g->voxsize.y;
  cellmax.z = cellmin.z + g->voxsize.z;

  cmin->x =  FHUGE;   cmin->y =  FHUGE;   cmin->z =  FHUGE;
  cmax->x = -FHUGE;   cmax->y = -FHUGE;   cmax->z = -FHUGE;

  while (cur != NULL)  {  /* Go! */
    min.x = -FHUGE; min.y = -FHUGE; min.z = -FHUGE;
    max.x =  FHUGE; max.y =  FHUGE; max.z =  FHUGE;

    if (cur->obj->methods->bbox((void *) cur->obj, &min, &max)) {
      if ((min.x >= cellmin.x) && (max.x <= cellmax.x) &&
          (min.y >= cellmin.y) && (max.y <= cellmax.y) &&
          (min.z >= cellmin.z) && (max.z <= cellmax.z)) {
      
        cmin->x = MYMIN( cmin->x , min.x);
        cmin->y = MYMIN( cmin->y , min.y);
        cmin->z = MYMIN( cmin->z , min.z);

        cmax->x = MYMAX( cmax->x , max.x);
        cmax->y = MYMAX( cmax->y , max.y);
        cmax->z = MYMAX( cmax->z , max.z);
      
        numinbounds++;
      }
    }

    cur=cur->next;
  }
 
  /* in case we get a 0.0 sized axis on the cell bounds, we'll */
  /* use the original cell bounds */
  if ((cmax->x - cmin->x) < EPSILON) {
    cmax->x += EPSILON;
    cmin->x -= EPSILON;
  }
  if ((cmax->y - cmin->y) < EPSILON) {
    cmax->y += EPSILON;
    cmin->y -= EPSILON;
  }
  if ((cmax->z - cmin->z) < EPSILON) {
    cmax->z += EPSILON;
    cmin->z -= EPSILON;
  }

  return numinbounds;
}

static int countobj(object * root) {
  object * cur;     /* counts the number of objects on a list */
  int numobj;

  numobj=0;
  cur=root;

  while (cur != NULL) {
    cur=(object *)cur->nextobj;
    numobj++;
  }
  return numobj;
}

static int countobjlist(objectlist * root) {
  objectlist * cur;
  int numobj;

  numobj=0; 
  cur = root;

  while (cur != NULL) {
    cur = cur->next;
    numobj++;
  }
  return numobj;
}

int engrid_scene(object ** list) {
  grid * g;
  int numobj, numcbrt;
  vector gmin, gmax;
  gridindex index;
 
  if (*list == NULL)
    return 0;

  numobj = countobj(*list);

  if ( !silent_mode )
    fprintf(stderr, "Scene contains %d bounded objects.\n", numobj);

  if (numobj > 16) {
    numcbrt = (int) cbrt(4*numobj);
    globalbound(list, &gmin, &gmax);

    g = (grid *) newgrid(numcbrt, numcbrt, numcbrt, gmin, gmax);
    engrid_objlist(g, list);

    numobj = countobj(*list);
    g->nextobj = *list;
    *list = (object *) g;

    /* now create subgrids.. */
    for (index.z=0; index.z<g->zsize; index.z++) {
      for (index.y=0; index.y<g->ysize; index.y++) {
        for (index.x=0; index.x<g->xsize; index.x++) {
          engrid_cell(g, &index);
        }
      }
    } 
  }

  return 1;
}


void engrid_objlist(grid * g, object ** list) {
  object * cur, * next, **prev;

  if (*list == NULL) 
    return;
  
  prev = list; 
  cur = *list;

  while (cur != NULL) {
    next = (object *)cur->nextobj;

    if (engrid_object(g, cur)) 
      *prev = next;
    else 
      prev = (object **) &cur->nextobj;

    cur = next;
  } 
}

static int engrid_cell(grid * gold, gridindex *index) {
  vector gmin, gmax, gsize;
  flt len;
  int numobj, numcbrt, xs, ys, zs;
  grid * g;
  objectlist **list;
  objectlist * newobj;

  list = &gold->cells[index->z*gold->xsize*gold->ysize + 
                     index->y*gold->xsize  + index->x];

  if (*list == NULL)
    return 0;

  numobj =  cellbound(gold, index, &gmin, &gmax);

  VSub(&gmax, &gmin, &gsize);
  len = 1.0 / (MYMAX( MYMAX(gsize.x, gsize.y), gsize.z ));
  gsize.x *= len;  
  gsize.y *= len;  
  gsize.z *= len;  

  if (numobj > 16) {
    numcbrt = (int) cbrt(2*numobj); 
    
    xs = (int) ((flt) numcbrt * gsize.x);
    if (xs < 1) xs = 1;
    ys = (int) ((flt) numcbrt * gsize.y);
    if (ys < 1) ys = 1;
    zs = (int) ((flt) numcbrt * gsize.z);
    if (zs < 1) zs = 1;

    g = (grid *) newgrid(xs, ys, zs, gmin, gmax);
    engrid_objectlist(g, list);

    newobj = (objectlist *) rt_getmem(sizeof(objectlist));    
    newobj->obj = (object *) g;
    newobj->next = *list;
    *list = newobj;

    g->nextobj = gold->objects;
    gold->objects = (object *) g;
  }

  return 1;
}

static int engrid_objectlist(grid * g, objectlist ** list) {
  objectlist * cur, * next, **prev;
  int numsucceeded = 0; 

  if (*list == NULL) 
    return 0;
  
  prev = list; 
  cur = *list;

  while (cur != NULL) {
    next = cur->next;

    if (engrid_object(g, cur->obj)) {
      *prev = next;
      free(cur);
      numsucceeded++;
    }
    else {
      prev = &cur->next;
    }

    cur = next;
  } 

  return numsucceeded;
}



static int engrid_object(grid * g, object * obj) {
  vector omin, omax; 
  gridindex low, high;
  int x, y, z, zindex, yindex, voxindex;
  objectlist * tmp;
 
  if (obj->methods->bbox(obj, &omin, &omax)) { 
    if (!pos2grid(g, &omin, &low) || !pos2grid(g, &omax, &high)) {
      return 0; /* object is not wholly contained in the grid */
    }
  }
  else {
    return 0; /* object is unbounded */
  }

  /* add the object to the complete list of objects in the grid */
  obj->nextobj = g->objects;
  g->objects = obj;

  /* add this object to all voxels it inhabits */
  for (z=low.z; z<=high.z; z++) {
    zindex = z * g->xsize * g->ysize;
    for (y=low.y; y<=high.y; y++) {
      yindex = y * g->xsize;
      for (x=low.x; x<=high.x; x++) {
        voxindex = x + yindex + zindex; 
        tmp = (objectlist *) rt_getmem(sizeof(objectlist));
        tmp->next = g->cells[voxindex];
        tmp->obj = obj;
        g->cells[voxindex] = tmp;
      }
    }
  }
 
  return 1;
}

static int pos2grid(grid * g, vector * pos, gridindex * index) {
  index->x = (int) ((pos->x - g->min.x) / g->voxsize.x);
  index->y = (int) ((pos->y - g->min.y) / g->voxsize.y);
  index->z = (int) ((pos->z - g->min.z) / g->voxsize.z);
  
  if (index->x == g->xsize)
    index->x--;
  if (index->y == g->ysize)
    index->y--;
  if (index->z == g->zsize)
    index->z--;

  if (index->x < 0 || index->x > g->xsize ||
      index->y < 0 || index->y > g->ysize ||
      index->z < 0 || index->z > g->zsize) 
    return 0;

  if (pos->x < g->min.x || pos->x > g->max.x ||
      pos->y < g->min.y || pos->y > g->max.y ||
      pos->z < g->min.z || pos->z > g->max.z) 
    return 0; 

  return 1;
}


/* the real thing */
static void grid_intersect(grid * g, ray * ry) {
  flt tnear, tfar, offset;
  vector curpos, tmax, tdelta, pdeltaX, pdeltaY, pdeltaZ, nXp, nYp, nZp;
  gridindex curvox, step, out; 
  int voxindex;
  objectlist * cur;

  if (ry->flags & RT_RAY_FINISHED)
    return;

  if (!grid_bounds_intersect(g, ry, &tnear, &tfar))
    return;
 
  if (ry->maxdist < tnear)
    return;
 
  curpos = Raypnt(ry, tnear); 
  pos2grid(g, &curpos, &curvox);
  offset = tnear;

  /* Setup X iterator stuff */
  if (fabs(ry->d.x) < EPSILON) {
    tmax.x = FHUGE;
    tdelta.x = 0.0;
    step.x = 0;
    out.x = 0; /* never goes out of bounds on this axis */
  }
  else if (ry->d.x < 0.0) {
    tmax.x = offset + ((voxel2x(g, curvox.x) - curpos.x) / ry->d.x); 
    tdelta.x = g->voxsize.x / - ry->d.x;
    step.x = out.x = -1;
  }
  else {
    tmax.x = offset + ((voxel2x(g, curvox.x + 1) - curpos.x) / ry->d.x);
    tdelta.x = g->voxsize.x / ry->d.x;
    step.x = 1;
    out.x = g->xsize;
  }

  /* Setup Y iterator stuff */
  if (fabs(ry->d.y) < EPSILON) {
    tmax.y = FHUGE;
    tdelta.y = 0.0; 
    step.y = 0;
    out.y = 0; /* never goes out of bounds on this axis */
  }
  else if (ry->d.y < 0.0) {
    tmax.y = offset + ((voxel2y(g, curvox.y) - curpos.y) / ry->d.y);
    tdelta.y = g->voxsize.y / - ry->d.y;
    step.y = out.y = -1;
  }
  else {
    tmax.y = offset + ((voxel2y(g, curvox.y + 1) - curpos.y) / ry->d.y);
    tdelta.y = g->voxsize.y / ry->d.y;
    step.y = 1;
    out.y = g->ysize;
  }

  /* Setup Z iterator stuff */
  if (fabs(ry->d.z) < EPSILON) {
    tmax.z = FHUGE;
    tdelta.z = 0.0; 
    step.z = 0;
    out.z = 0; /* never goes out of bounds on this axis */
  }
  else if (ry->d.z < 0.0) {
    tmax.z = offset + ((voxel2z(g, curvox.z) - curpos.z) / ry->d.z);
    tdelta.z = g->voxsize.z / - ry->d.z;
    step.z = out.z = -1;
  }
  else {
    tmax.z = offset + ((voxel2z(g, curvox.z + 1) - curpos.z) / ry->d.z);
    tdelta.z = g->voxsize.z / ry->d.z;
    step.z = 1;
    out.z = g->zsize;
  }

  pdeltaX = ry->d;
  VScale(&pdeltaX, tdelta.x);
  pdeltaY = ry->d;
  VScale(&pdeltaY, tdelta.y);
  pdeltaZ = ry->d;
  VScale(&pdeltaZ, tdelta.z);

  nXp = Raypnt(ry, tmax.x);
  nYp = Raypnt(ry, tmax.y);
  nZp = Raypnt(ry, tmax.z);

  voxindex = curvox.z*g->xsize*g->ysize + curvox.y*g->xsize + curvox.x; 
  while (1) {
    if (tmax.x < tmax.y && tmax.x < tmax.z) {
      cur = g->cells[voxindex];
      while (cur != NULL) {
        if (ry->mbox[cur->obj->id] != ry->serial) {
          ry->mbox[cur->obj->id] = ry->serial; 
          cur->obj->methods->intersect(cur->obj, ry);
        }
        cur = cur->next;
      }
      curvox.x += step.x;
      if (ry->maxdist < tmax.x || curvox.x == out.x) 
        break; 
      voxindex += step.x;
      tmax.x += tdelta.x;
      curpos = nXp;
      nXp.x += pdeltaX.x;
      nXp.y += pdeltaX.y;
      nXp.z += pdeltaX.z;
    }
    else if (tmax.z < tmax.y) {
      cur = g->cells[voxindex];
      while (cur != NULL) {
        if (ry->mbox[cur->obj->id] != ry->serial) {
          ry->mbox[cur->obj->id] = ry->serial; 
          cur->obj->methods->intersect(cur->obj, ry);
        }
        cur = cur->next;
      }
      curvox.z += step.z;
      if (ry->maxdist < tmax.z || curvox.z == out.z) 
        break;
      voxindex += step.z*g->xsize*g->ysize;
      tmax.z += tdelta.z;
      curpos = nZp;
      nZp.x += pdeltaZ.x;
      nZp.y += pdeltaZ.y;
      nZp.z += pdeltaZ.z;
    }
    else {
      cur = g->cells[voxindex];
      while (cur != NULL) {
        if (ry->mbox[cur->obj->id] != ry->serial) {
          ry->mbox[cur->obj->id] = ry->serial; 
          cur->obj->methods->intersect(cur->obj, ry);
        }
        cur = cur->next;
      }
      curvox.y += step.y;
      if (ry->maxdist < tmax.y || curvox.y == out.y) 
        break;
      voxindex += step.y*g->xsize;
      tmax.y += tdelta.y;
      curpos = nYp;
      nYp.x += pdeltaY.x;
      nYp.y += pdeltaY.y;
      nYp.z += pdeltaY.z;
    }

    if (ry->flags & RT_RAY_FINISHED)
      break;
  }
}

static void voxel_intersect(grid * g, ray * ry, int voxindex) {
  objectlist * cur;

  cur = g->cells[voxindex];
  while (cur != NULL) {
    cur->obj->methods->intersect(cur->obj, ry);
    cur = cur->next;
  }
}

static int grid_bounds_intersect(grid * g, ray * ry, flt *nr, flt *fr) {
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;

  tnear= -FHUGE;
  tfar= FHUGE;

  if (ry->d.x == 0.0) {
    if ((ry->o.x < g->min.x) || (ry->o.x > g->max.x)) return 0;
  }
  else {
    tx1 = (g->min.x - ry->o.x) / ry->d.x;
    tx2 = (g->max.x - ry->o.x) / ry->d.x;
    if (tx1 > tx2) { a=tx1; tx1=tx2; tx2=a; }
    if (tx1 > tnear) tnear=tx1;
    if (tx2 < tfar)   tfar=tx2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  if (ry->d.y == 0.0) {
    if ((ry->o.y < g->min.y) || (ry->o.y > g->max.y)) return 0;
  }
  else {
    ty1 = (g->min.y - ry->o.y) / ry->d.y;
    ty2 = (g->max.y - ry->o.y) / ry->d.y;
    if (ty1 > ty2) { a=ty1; ty1=ty2; ty2=a; }
    if (ty1 > tnear) tnear=ty1;
    if (ty2 < tfar)   tfar=ty2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  if (ry->d.z == 0.0) {
    if ((ry->o.z < g->min.z) || (ry->o.z > g->max.z)) return 0;
  }
  else {
    tz1 = (g->min.z - ry->o.z) / ry->d.z;
    tz2 = (g->max.z - ry->o.z) / ry->d.z;
    if (tz1 > tz2) { a=tz1; tz1=tz2; tz2=a; }
    if (tz1 > tnear) tnear=tz1;
    if (tz2 < tfar)   tfar=tz2;
  }
  if (tnear > tfar) return 0;
  if (tfar < 0.0) return 0;

  *nr = tnear;
  *fr = tfar; 
  return 1;
}
