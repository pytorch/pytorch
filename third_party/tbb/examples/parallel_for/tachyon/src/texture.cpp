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
 * texture.cpp - This file contains functions for implementing textures.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "texture.h"
#include "coordsys.h"
#include "imap.h"
#include "vector.h"
#include "box.h"

/* plain vanilla texture solely based on object color */
color standard_texture(vector * hit, texture * tex, ray * ry) {
  return tex->col;
}

/* cylindrical image map */
color image_cyl_texture(vector * hit, texture * tex, ray * ry) {
  vector rh;
  flt u,v;
 
  rh.x=hit->x - tex->ctr.x;
  rh.z=hit->y - tex->ctr.y;
  rh.y=hit->z - tex->ctr.z;
 
  xyztocyl(rh, 1.0, &u, &v);

  u = u * tex->scale.x;  
  u = u + tex->rot.x;
  u=fmod(u, 1.0);
  if (u < 0.0) u+=1.0; 

  v = v * tex->scale.y; 
  v = v + tex->rot.y;
  v=fmod(v, 1.0);
  if (v < 0.0) v+=1.0; 

  return ImageMap((rawimage *)tex->img, u, v); 
}  

/* spherical image map */
color image_sphere_texture(vector * hit, texture * tex, ray * ry) {
  vector rh;
  flt u,v;
 
  rh.x=hit->x - tex->ctr.x;
  rh.y=hit->y - tex->ctr.y;
  rh.z=hit->z - tex->ctr.z;
 
  xyztospr(rh, &u, &v);

  u = u * tex->scale.x;
  u = u + tex->rot.x;
  u=fmod(u, 1.0);
  if (u < 0.0) u+=1.0;
 
  v = v * tex->scale.y;
  v = v + tex->rot.y;
  v=fmod(v, 1.0);
  if (v < 0.0) v+=1.0;
 
  return ImageMap((rawimage *)tex->img, u, v);
}

/* planar image map */
color image_plane_texture(vector * hit, texture * tex, ray * ry) {
  vector pnt;
  flt u,v;
 
  pnt.x=hit->x - tex->ctr.x;
  pnt.y=hit->y - tex->ctr.y;
  pnt.z=hit->z - tex->ctr.z;

  VDOT(u, tex->uaxs, pnt);
/*  VDOT(len, tex->uaxs, tex->uaxs);
  u = u / sqrt(len); */

  VDOT(v, tex->vaxs, pnt); 
/*  VDOT(len, tex->vaxs, tex->vaxs);
  v = v / sqrt(len); */
    

  u = u * tex->scale.x;
  u = u + tex->rot.x;
  u = fmod(u, 1.0);
  if (u < 0.0) u += 1.0;

  v = v * tex->scale.y;
  v = v + tex->rot.y;
  v = fmod(v, 1.0);
  if (v < 0.0) v += 1.0;

  return ImageMap((rawimage *)tex->img, u, v);
}

color grit_texture(vector * hit, texture * tex, ray * ry) {
  int rnum;
  flt fnum;
  color col;

  rnum=rand() % 4096;
  fnum=(rnum / 4096.0 * 0.2) + 0.8;

  col.r=tex->col.r * fnum;
  col.g=tex->col.g * fnum;
  col.b=tex->col.b * fnum;

  return col;
}

color checker_texture(vector * hit, texture * tex, ray * ry) {
  long x,y,z;
  flt xh,yh,zh;
  color col;

  xh=hit->x - tex->ctr.x; 
  x=(long) ((fabs(xh) * 3) + 0.5);
  x=x % 2;
  yh=hit->y - tex->ctr.y;
  y=(long) ((fabs(yh) * 3) + 0.5);
  y=y % 2;
  zh=hit->z - tex->ctr.z;
  z=(long) ((fabs(zh) * 3) + 0.5);
  z=z % 2;

  if (((x + y + z) % 2)==1) {
    col.r=1.0;
    col.g=0.2;
    col.b=0.0;
  }
  else {
    col.r=0.0;
    col.g=0.2;
    col.b=1.0;
  }

  return col;
}

color cyl_checker_texture(vector * hit, texture * tex, ray * ry) {
  long x,y;
  vector rh;
  flt u,v;
  color col;
 
  rh.x=hit->x - tex->ctr.x;
  rh.y=hit->y - tex->ctr.y;
  rh.z=hit->z - tex->ctr.z;

  xyztocyl(rh, 1.0, &u, &v); 

  x=(long) (fabs(u) * 18.0);
  x=x % 2;
  y=(long) (fabs(v) * 10.0);
  y=y % 2;
 
  if (((x + y) % 2)==1) {
    col.r=1.0;
    col.g=0.2;
    col.b=0.0;
  }
  else {
    col.r=0.0;
    col.g=0.2;
    col.b=1.0;
  }
 
  return col;
}


color wood_texture(vector * hit, texture * tex, ray * ry) {
  flt radius, angle;
  int grain;
  color col;
  flt x,y,z;

  x=(hit->x - tex->ctr.x) * 1000;
  y=(hit->y - tex->ctr.y) * 1000;
  z=(hit->z - tex->ctr.z) * 1000;

  radius=sqrt(x*x + z*z);
  if (z == 0.0) 
    angle=3.1415926/2.0;
  else 
    angle=atan(x / z);

  radius=radius + 3.0 * sin(20 * angle + y / 150.0);
  grain=((int) (radius + 0.5)) % 60;
  if (grain < 40) {
    col.r=0.8;
    col.g=1.0;
    col.b=0.2;
  }
  else {
    col.r=0.0;
    col.g=0.0;
    col.b=0.0;
  }     

  return col;
} 



#define NMAX 28
short int NoiseMatrix[NMAX][NMAX][NMAX];

void InitNoise(void) {
  byte x,y,z,i,j,k;

  for (x=0; x<NMAX; x++) {
    for (y=0; y<NMAX; y++) {
      for (z=0; z<NMAX; z++) {
        NoiseMatrix[x][y][z]=rand() % 12000;

        if (x==NMAX-1) i=0; 
        else i=x;

        if (y==NMAX-1) j=0;
        else j=y;

        if (z==NMAX-1) k=0;
        else k=z;

        NoiseMatrix[x][y][z]=NoiseMatrix[i][j][k];
      }
    }
  }
}

int Noise(flt x, flt y, flt z) {
  byte ix, iy, iz;
  flt ox, oy, oz;
  int p000, p001, p010, p011;
  int p100, p101, p110, p111;
  int p00, p01, p10, p11;
  int p0, p1;
  int d00, d01, d10, d11;
  int d0, d1, d;

  x=fabs(x);
  y=fabs(y);
  z=fabs(z);

  ix=((int) x) % (NMAX-1);
  iy=((int) y) % (NMAX-1);
  iz=((int) z) % (NMAX-1);

  ox=(x - ((int) x));
  oy=(y - ((int) y));
  oz=(z - ((int) z));

  p000=NoiseMatrix[ix][iy][iz];
  p001=NoiseMatrix[ix][iy][iz+1];
  p010=NoiseMatrix[ix][iy+1][iz];
  p011=NoiseMatrix[ix][iy+1][iz+1];
  p100=NoiseMatrix[ix+1][iy][iz];
  p101=NoiseMatrix[ix+1][iy][iz+1];
  p110=NoiseMatrix[ix+1][iy+1][iz];
  p111=NoiseMatrix[ix+1][iy+1][iz+1];

  d00=p100-p000;
  d01=p101-p001;
  d10=p110-p010;
  d11=p111-p011;

  p00=(int) ((int) d00*ox) + p000;
  p01=(int) ((int) d01*ox) + p001;
  p10=(int) ((int) d10*ox) + p010;
  p11=(int) ((int) d11*ox) + p011;
  d0=p10-p00;
  d1=p11-p01;
  p0=(int) ((int) d0*oy) + p00;
  p1=(int) ((int) d1*oy) + p01;
  d=p1-p0;

  return (int) ((int) d*oz) + p0;
}

color marble_texture(vector * hit, texture * tex, ray * ry) {
  flt i,d;
  flt x,y,z;
  color col;
 
  x=hit->x;
  y=hit->y; 
  z=hit->z;

  x=x * 1.0;

  d=x + 0.0006 * Noise(x, (y * 1.0), (z * 1.0));
  d=d*(((int) d) % 25);
  i=0.0 + 0.10 * fabs(d - 10.0 - 20.0 * ((int) d * 0.05));
  if (i > 1.0) i=1.0;
  if (i < 0.0) i=0.0;  

/*
  col.r=i * tex->col.r;
  col.g=i * tex->col.g;
  col.b=i * tex->col.b;
*/

  col.r = (1.0 + sin(i * 6.28)) / 2.0;
  col.g = (1.0 + sin(i * 16.28)) / 2.0;
  col.b = (1.0 + cos(i * 30.28)) / 2.0;

  return col;      
}


color gnoise_texture(vector * hit, texture * tex, ray * ry) {
  color col;
  flt f;

  f=Noise((hit->x - tex->ctr.x), 
          (hit->y - tex->ctr.y), 
	  (hit->z - tex->ctr.z));

  if (f < 0.01) f=0.01;
  if (f > 1.0) f=1.0;

  col.r=tex->col.r * f;
  col.g=tex->col.g * f;
  col.b=tex->col.b * f;

  return col;
}

void InitTextures(void) {
  InitNoise();
  ResetImages();
}

