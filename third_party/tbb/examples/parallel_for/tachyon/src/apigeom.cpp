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
 * api.cpp - This file contains all of the API calls that are defined for
 *         external driver code to use.  
 */

#include "machine.h"
#include "types.h"
#include "api.h"
#include "macros.h"
#include "vector.h"

#define MyVNorm(a)		VNorm ((vector *) a)

void rt_polycylinder(void * tex, vector * points, int numpts, apiflt rad) {
  vector a;
  int i;

  if ((points == NULL) || (numpts == 0)) {
    return;
  }

  if (numpts > 0) {
    rt_sphere(tex, points[0], rad);
    
    if (numpts > 1) {
      for (i=1; i<numpts; i++) {
        a.x = points[i].x - points[i-1].x;
        a.y = points[i].y - points[i-1].y;
        a.z = points[i].z - points[i-1].z;
        
        rt_fcylinder(tex, points[i-1], a, rad);
        rt_sphere(tex, points[i], rad);
      }
    }
  }
}

void rt_heightfield(void * tex, vector ctr, int m, int n, 
                    apiflt * field, apiflt wx, apiflt wy) {
  int xx,yy; 
  vector v0, v1, v2; 
  apiflt xoff, yoff, zoff;

  xoff=ctr.x - (wx / 2.0);
  yoff=ctr.z - (wy / 2.0);
  zoff=ctr.y;

  for (yy=0; yy<(n-1); yy++) { 
    for (xx=0; xx<(m-1); xx++) {
      v0.x=wx*(xx    )/(m*1.0) + xoff; 
      v0.y=field[(yy    )*m + (xx    )] + zoff;
      v0.z=wy*(yy    )/(n*1.0) + yoff;

      v1.x=wx*(xx + 1)/(m*1.0) + xoff; 
      v1.y=field[(yy    )*m + (xx + 1)] + zoff;
      v1.z=wy*(yy    )/(n*1.0) + yoff;

      v2.x=wx*(xx + 1)/(m*1.0) + xoff; 
      v2.y=field[(yy + 1)*m + (xx + 1)] + zoff;
      v2.z=wy*(yy + 1)/(n*1.0) + yoff;

      rt_tri(tex, v1, v0, v2);

      v0.x=wx*(xx    )/(m*1.0) + xoff;
      v0.y=field[(yy    )*m + (xx    )] + zoff;
      v0.z=wy*(yy    )/(n*1.0) + yoff;

      v1.x=wx*(xx    )/(m*1.0) + xoff;
      v1.y=field[(yy + 1)*m + (xx    )] + zoff;
      v1.z=wy*(yy + 1)/(n*1.0) + yoff;

      v2.x=wx*(xx + 1)/(m*1.0) + xoff;
      v2.y=field[(yy + 1)*m + (xx + 1)] + zoff;
      v2.z=wy*(yy + 1)/(n*1.0) + yoff;
 
      rt_tri(tex, v0, v1, v2);
    }
  } 
} /* end of heightfield */


static void rt_sheightfield(void * tex, vector ctr, int m, int n, 
                    apiflt * field, apiflt wx, apiflt wy) {
  vector * vertices;
  vector * normals;
  vector offset;
  apiflt xinc, yinc;
  int x, y, addr; 
   
  vertices = (vector *) malloc(m*n*sizeof(vector));
  normals = (vector *) malloc(m*n*sizeof(vector));

  offset.x = ctr.x - (wx / 2.0);
  offset.y = ctr.z - (wy / 2.0);
  offset.z = ctr.y;

  xinc = wx / ((apiflt) m);
  yinc = wy / ((apiflt) n);

  /* build vertex list */
  for (y=0; y<n; y++) { 
    for (x=0; x<m; x++) {
      addr = y*m + x;
      vertices[addr] = rt_vector(
        x * xinc + offset.x,
        field[addr] + offset.z,
        y * yinc + offset.y);
    }
  }

  /* build normals from vertex list */
  for (x=1; x<m; x++) {
    normals[x] = normals[(n - 1)*m + x] = rt_vector(0.0, 1.0, 0.0);
  }
  for (y=1; y<n; y++) {
    normals[y*m] = normals[y*m + (m-1)] = rt_vector(0.0, 1.0, 0.0);
  }
  for (y=1; y<(n-1); y++) {
    for (x=1; x<(m-1); x++) {
      addr = y*m + x;

      normals[addr] = rt_vector(
        -(field[addr + 1] - field[addr - 1]) / (2.0 * xinc), 
        1.0, 
        -(field[addr + m] - field[addr - m]) / (2.0 * yinc));

      MyVNorm(&normals[addr]);
    }
  }    

  /* generate actual triangles */
  for (y=0; y<(n-1); y++) {
    for (x=0; x<(m-1); x++) {
      addr = y*m + x;

      rt_stri(tex, vertices[addr], vertices[addr + 1 + m], vertices[addr + 1],
                   normals[addr], normals[addr + 1 + m], normals[addr + 1]);
      rt_stri(tex, vertices[addr], vertices[addr + m], vertices[addr + 1 + m],
                   normals[addr], normals[addr + m], normals[addr + 1 + m]);
    }
  }

  free(normals);
  free(vertices);
} /* end of smoothed heightfield */


static void adjust(apiflt *base, int xres, int yres, apiflt wx, apiflt wy, 
		int xa, int ya, int x, int y, int xb, int yb) {
  apiflt d, v;
  
  if (base[x + (xres*y)]==0.0) { 

    d=(abs(xa - xb) / (xres * 1.0))*wx + (abs(ya - yb) / (yres * 1.0))*wy; 

    v=(base[xa + (xres*ya)] + base[xb + (xres*yb)]) / 2.0 +
       (((((rand() % 1000) - 500.0)/500.0)*d) / 8.0);

    if (v < 0.0) v=0.0; 
    if (v > (xres + yres)) v=(xres + yres);
    base[x + (xres * y)]=v; 
 } 
}

static void subdivide(apiflt *base, int xres, int yres, apiflt wx, apiflt wy,
                  int x1, int y1, int x2, int y2) {
  long x,y;

  if (((x2 - x1) < 2) && ((y2 - y1) < 2)) { return; }

  x=(x1 + x2) / 2;
  y=(y1 + y2) / 2;

  adjust(base, xres, yres, wx, wy, x1, y1, x, y1, x2, y1);
  adjust(base, xres, yres, wx, wy, x2, y1, x2, y, x2, y2);
  adjust(base, xres, yres, wx, wy, x1, y2, x, y2, x2, y2);
  adjust(base, xres, yres, wx, wy, x1, y1, x1, y, x1, y2);

 
  if (base[x + xres*y]==0.0) {
    base[x + (xres * y)]=(base[x1 + xres*y1] + base[x2 + xres*y1] +
                          base[x2 + xres*y2] + base[x1 + xres*y2]   )/4.0;
  }
 
  subdivide(base, xres, yres, wx, wy, x1, y1 ,x ,y);
  subdivide(base, xres, yres, wx, wy, x, y1, x2, y);
  subdivide(base, xres, yres, wx, wy, x, y, x2, y2);
  subdivide(base, xres, yres, wx, wy, x1, y, x, y2);
}

void rt_landscape(void * tex, int m, int n, 
              	vector ctr, apiflt wx, apiflt wy) {
  int totalsize, x, y;
  apiflt * field; 

  totalsize=m*n;

  srand(totalsize);

  field=(apiflt *) malloc(totalsize*sizeof(apiflt));

  for (y=0; y<n; y++) {
    for (x=0; x<m; x++) {
       field[x + y*m]=0.0;
    }
  }

  field[0 + 0]=1.0 + (rand() % 100)/100.0;
  field[m - 1]=1.0 + (rand() % 100)/100.0;
  field[0     + m*(n - 1)]=1.0 + (rand() % 100)/100.0;
  field[m - 1 + m*(n - 1)]=1.0 + (rand() % 100)/100.0;

  subdivide(field, m, n, wx, wy, 0, 0, m-1, n-1);

  rt_sheightfield(tex, ctr, m, n, field, wx, wy);

  free(field);
}

