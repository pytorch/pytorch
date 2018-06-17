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
 * vector.cpp - This file contains all of the vector arithmetic functions.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"

flt VDot(vector *a, vector *b) {
  return (a->x*b->x + a->y*b->y + a->z*b->z);
}

void VCross(vector * a, vector * b, vector * c) {
  c->x = (a->y * b->z) - (a->z * b->y);
  c->y = (a->z * b->x) - (a->x * b->z);
  c->z = (a->x * b->y) - (a->y * b->x);
}

flt VLength(vector * a) {
  return (flt) sqrt((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
}

void VNorm(vector * a) {
  flt len;

  len=sqrt((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
  if (len != 0.0) {
    a->x /= len;
    a->y /= len;
    a->z /= len;
  }
}

void VAdd(vector * a, vector * b, vector * c) {
  c->x = (a->x + b->x);
  c->y = (a->y + b->y);
  c->z = (a->z + b->z);
}
    
void VSub(vector * a, vector * b, vector * c) {
  c->x = (a->x - b->x);
  c->y = (a->y - b->y);
  c->z = (a->z - b->z);
}

void VAddS(flt a, vector * A, vector * B, vector * C) {
  C->x = (a * A->x) + B->x;
  C->y = (a * A->y) + B->y;
  C->z = (a * A->z) + B->z;
}

vector Raypnt(ray * a, flt t) {
  vector temp;

  temp.x=a->o.x + (a->d.x * t);
  temp.y=a->o.y + (a->d.y * t);
  temp.z=a->o.z + (a->d.z * t);

  return temp;
}

void VScale(vector * a, flt s) {
  a->x *= s;
  a->y *= s;
  a->z *= s;
}

void ColorAddS(color * a, color * b, flt s) {
  a->r += b->r * s;
  a->g += b->g * s;
  a->b += b->b * s;
}

void ColorAccum(color * a, color * b) {
  a->r += b->r;
  a->g += b->g;
  a->b += b->b;
}

void ColorScale(color * a, flt s) {
  a->r *= s;
  a->g *= s;
  a->b *= s;
}

