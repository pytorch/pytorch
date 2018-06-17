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
 * grid.h - spatial subdivision efficiency structures
 *
 * $Id: grid.h,v 1.2 2007-02-22 17:54:15 Exp $
 * 
 */

int engrid_scene(object ** list);
object * newgrid(int xsize, int ysize, int zsize, vector min, vector max);

#ifdef GRID_PRIVATE

typedef struct objectlist {
  struct objectlist * next; /* next link in the list */
  object * obj;             /* the actual object     */
} objectlist; 

typedef struct {
  unsigned int id;                      /* Unique Object serial number    */
  void * nextobj;                       /* pointer to next object in list */
  object_methods * methods;             /* this object's methods          */
  texture * tex;                        /* object texture                 */
  int xsize;           /* number of cells along the X direction */
  int ysize;           /* number of cells along the Y direction */
  int zsize;           /* number of cells along the Z direction */
  vector min;          /* the minimum coords for the box containing the grid */
  vector max;          /* the maximum coords for the box containing the grid */
  vector voxsize;      /* the size of a grid cell/voxel */
  object * objects;    /* all objects contained in the grid */
  objectlist ** cells; /* the grid cells themselves */
} grid;

typedef struct {
  int x;         /* Voxel X address */
  int y;         /* Voxel Y address */
  int z;         /* Voxel Z address */
} gridindex; 

/*
 * Convert from voxel number along X/Y/Z to corresponding coordinate.
 */
#define voxel2x(g,X)  ((X) * (g->voxsize.x) + (g->min.x))
#define voxel2y(g,Y)  ((Y) * (g->voxsize.y) + (g->min.y))
#define voxel2z(g,Z)  ((Z) * (g->voxsize.z) + (g->min.z))

/*
 * And vice-versa.
 */
#define x2voxel(g,x)            (((x) - g->min.x) / g->voxsize.x)
#define y2voxel(g,y)            (((y) - g->min.y) / g->voxsize.y)
#define z2voxel(g,z)            (((z) - g->min.z) / g->voxsize.z)


static int grid_bbox(void * obj, vector * min, vector * max);
static void grid_free(void * v);

static int cellbound(grid *g, gridindex *index, vector * cmin, vector * cmax);

void engrid_objlist(grid * g, object ** list);
static int engrid_object(grid * g, object * obj);

static int engrid_objectlist(grid * g, objectlist ** list);
static int engrid_cell(grid *, gridindex *);

static int pos2grid(grid * g, vector * pos, gridindex * index);
static void grid_intersect(grid *, ray *);
static void voxel_intersect(grid * g, ray * ry, int voxaddr);
static int grid_bounds_intersect(grid * g, ray * ry, flt *near, flt *far); 


#endif
