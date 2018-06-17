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
 *  imageio.cpp - This file deals with reading/writing image files
 */ 

/* For our puposes, we're interested only in the 3 byte per pixel 24 bit
 * truecolor sort of file..
 */

#include <stdio.h>
#include "machine.h"
#include "types.h"
#include "util.h"
#include "imageio.h"
#include "ppm.h"     /* PPM files */
#include "tgafile.h" /* Truevision Targa files */
#include "jpeg.h"    /* JPEG files */

static 
int fakeimage(char * name, int * xres, int * yres, unsigned char ** imgdata) {
  int i, imgsize;

  fprintf(stderr, "Error loading image %s.  Faking it.\n", name);
   
  *xres = 2;
  *yres = 2;
  imgsize = 3 * (*xres) * (*yres);
  *imgdata = (unsigned char *)rt_getmem(imgsize);
  for (i=0; i<imgsize; i++) {
    (*imgdata)[i] = 255;
  }

  return IMAGENOERR;
}


int readimage(rawimage * img) {
  int rc;
  int xres, yres;
  unsigned char * imgdata = NULL;
  char * name = img->name;

  if (strstr(name, ".ppm")) { 
    rc = readppm(name, &xres, &yres, &imgdata);
  }
  else if (strstr(name, ".tga")) {
    rc = readtga(name, &xres, &yres, &imgdata);
  }
  else if (strstr(name, ".jpg")) {
    rc = readjpeg(name, &xres, &yres, &imgdata);
  }
  else if (strstr(name, ".gif")) {
    rc = IMAGEUNSUP; 
  }
  else if (strstr(name, ".png")) {
    rc = IMAGEUNSUP; 
  }
  else if (strstr(name, ".tiff")) {
    rc = IMAGEUNSUP; 
  }
  else if (strstr(name, ".rgb")) {
    rc = IMAGEUNSUP; 
  }
  else if (strstr(name, ".xpm")) {
    rc = IMAGEUNSUP; 
  }
  else {
    rc = readppm(name, &xres, &yres, &imgdata);
  } 

  switch (rc) {
    case IMAGEREADERR:
      fprintf(stderr, "Short read encountered while loading image %s\n", name);
      rc = IMAGENOERR; /* remap to non-fatal error */
      break;

    case IMAGEUNSUP:
      fprintf(stderr, "Cannot read unsupported image format for image %s\n", name);
      break;
  }    

  /* If the image load failed, create a tiny white colored image to fake it */ 
  /* this allows a scene to render even when a file can't be loaded */
  if (rc != IMAGENOERR) {
    rc = fakeimage(name, &xres, &yres, &imgdata);
  }

  /* If we succeeded in loading the image, return it. */
  if (rc == IMAGENOERR) { 
    img->xres = xres;
    img->yres = yres;
    img->bpp = 3;  
    img->data = imgdata;
  }

  return rc;
}
 

