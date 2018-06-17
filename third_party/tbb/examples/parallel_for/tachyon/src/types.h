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

#if __MINGW32__
#include <malloc.h>
#elif _WIN32
#include <malloc.h>
#define alloca _alloca
#elif __FreeBSD__||__NetBSD__
#include <stdlib.h>
#else
#include <alloca.h>
#endif

/* 
 * types.h - This file contains all of the type definitions for the raytracer
 *
 *  $Id: types.h,v 1.2 2007-02-22 17:54:16 Exp $
 */

#define MAXOCTNODES 25       /* subdivide octants /w > # of children */
#define SPEPSILON 0.000001   /* amount to crawl down a ray           */
#define EPSILON   0.000001   /* amount to crawl down a ray           */
#define TWOPI 6.2831853      /* guess                                */
#define FHUGE 1e18           /* biggest fp number we can represent   */

/* Maximum internal table sizes */
/* Use prime numbers for best memory system performance */
#define INTTBSIZE 1024       /* maximum intersections we can hold    */ 
#define MAXLIGHTS 39         /* maximum number of lights in a scene  */
#define MAXIMGS   39         /* maxiumum number of distinct images   */
#define RPCQSIZE  113	     /* number of RPC messages to queue      */

/* Parameter values for rt_boundmode() */
#define RT_BOUNDING_DISABLED 0  /* spatial subdivision/bounding disabled */
#define RT_BOUNDING_ENABLED  1  /* spatial subdivision/bounding enabled  */

/* Parameter values for rt_displaymode() */
#define RT_DISPLAY_DISABLED  0  /* video output enabled  */
#define RT_DISPLAY_ENABLED   1  /* video output disabled */

/* Ray flags */
#define RT_RAY_REGULAR   1
#define RT_RAY_SHADOW    2
#define RT_RAY_BOUNDED   4
#define RT_RAY_FINISHED  8

#ifdef USESINGLEFLT
typedef float flt;   /* generic floating point number, using float */
#else
typedef double flt;  /* generic floating point number, using double */
#endif

typedef unsigned char byte; /* 1 byte */
typedef signed int word;    /* 32 bit integer */

typedef struct {
   flt x;        /* X coordinate value */
   flt y;        /* Y coordinate value */
   flt z;        /* Z coordinate value */
} vector;

typedef struct {
   flt r;        /* Red component   */
   flt g;        /* Green component */
   flt b;        /* Blue component  */
} color;

typedef struct {
   byte r;       /* Red component   */
   byte g;       /* Green component */
   byte b;       /* Blue component  */
} bytecolor;

typedef struct {         /* Raw 24 bit image structure, for tga, ppm etc */
  int loaded;            /* image memory residence flag    */
  int xres;              /* image X axis size              */
  int yres;              /* image Y axis size              */
  int bpp;               /* image bits per pixel           */
  char name[96];         /* image filename (with path)     */
  unsigned char * data;  /* pointer to raw byte image data */
} rawimage;

typedef struct {         /* Scalar Volume Data */
  int loaded;            /* Volume data memory residence flag */
  int xres;		 /* volume X axis size                */
  int yres;		 /* volume Y axis size                */
  int zres;		 /* volume Z axis size                */
  flt opacity;		 /* opacity per unit length           */
  char name[96];         /* Volume data filename              */
  unsigned char * data;  /* pointer to raw byte volume data   */
} scalarvol;
 
typedef struct {
  color (* texfunc)(void *, void *, void *);
  int shadowcast;  /* does the object cast a shadow */
  int islight;	   /* light flag... */
  color col;       /* base object color */
  flt ambient;     /* ambient lighting */
  flt diffuse; 	   /* diffuse reflection */
  flt phong;       /* phong specular highlights */
  flt phongexp;    /* phong exponent/shininess factor */
  int phongtype;   /* phong type: 0 == plastic, nonzero == metal */
  flt specular;    /* specular reflection */
  flt opacity;     /* how opaque the object is */ 
  vector ctr;      /* origin of texture */
  vector rot;      /* rotation of texture about origin */
  vector scale;    /* scale of texture in x,y,z */
  vector uaxs;	   /* planar map U axis */
  vector vaxs;	   /* planar map V axis */
  void * img;      /* pointer to image for image mapping */
  void * obj;      /* object ptr, hack for volume shaders for now */
} texture;

typedef struct {
  void (* intersect)(void *, void *);              /* intersection func ptr  */
  void (* normal)(void *, void *, void *, void *); /* normal function ptr    */
  int (* bbox)(void *, vector *, vector *);        /* return the object bbox */
  void (* free)(void *);                           /* free the object        */
} object_methods;
 
typedef struct {
  unsigned int id;                      /* Unique Object serial number    */
  void * nextobj;                       /* pointer to next object in list */ 
  object_methods * methods;             /* this object's methods          */
  texture * tex;                        /* object texture                 */
} object; 

typedef struct {
  object * obj;  /* to object we hit                        */ 
  flt t;         /* distance along the ray to the hit point */
} intersection;

typedef struct {
  int num;                      /* number of intersections    */
  intersection closest;         /* closest intersection > 0.0 */
  intersection list[INTTBSIZE]; /* list of all intersections  */ 
} intersectstruct;

typedef struct {
  char outfilename[200];     /* name of the output image                */
  unsigned char * rawimage;  /* pointer to a raw rgb image to be stored */
  int hres;                  /* horizontal output image resolution      */
  int vres;                  /* vertical output image resolution        */
  flt aspectratio;           /* aspect ratio of output image            */
  int raydepth;              /* maximum recursion depth                 */
  int antialiasing;          /* number of antialiasing rays to fire     */
  int verbosemode;           /* verbose reporting flag                  */
  int boundmode;             /* automatic spatial subdivision flag      */
  int boundthresh;           /* threshold number of subobjects          */
  int displaymode;           /* run-time X11 display flag               */
  vector camcent;            /* center of the camera in world coords    */
  vector camviewvec;         /* view direction of the camera  (Z axis)  */
  vector camrightvec;        /* right axis for the camera     (X axis)  */
  vector camupvec;           /* up axis for the camera        (Y axis)  */
  flt camzoom;               /* zoom factor for the camera              */
  color background;          /* scene background color                  */
} scenedef;

typedef struct {
   intersectstruct * intstruct; /* ptr to thread's intersection data       */ 
   unsigned int depth;   /* levels left to recurse.. (maxdepth - curdepth) */
   unsigned int flags;   /* ray flags, any special treatment needed etc    */
   unsigned int serial;  /* serial number of the ray                       */
   unsigned int * mbox;  /* mailbox array for optimizing intersections     */
   vector o;             /* origin of the ray X,Y,Z                        */
   vector d;             /* normalized direction of the ray                */
   flt maxdist;          /* maximum distance to search for intersections   */
   vector s;		 /* startpoint of the ray (may differ from origin  */
   vector e;             /* endpoint of the ray if bounded                 */
   scenedef * scene;     /* pointer to the scene, for global parms such as */
                         /* background colors etc                          */
} ray;

typedef struct {
  int type;      /* RPC call type            */
  int from;      /* Sending processor        */
  int len;       /* length of parms in bytes */
  void * parms;  /* Parameters to RPC        */
} rpcmsg;
