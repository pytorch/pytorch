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
 *  ppm.cpp - This file deals with PPM format image files (reading/writing)
 */ 

/* For our puposes, we're interested only in the 3 byte per pixel 24 bit
   truecolor sort of file..  Probably won't implement any decent checking
   at this point, probably choke on things like the # comments.. */

// Try preventing lots of GCC warnings about ignored results of fscanf etc.
#if !__INTEL_COMPILER

#if __GNUC__<4 || __GNUC__==4 && __GNUC_MINOR__<5
// For older versions of GCC, disable use of __wur in GLIBC
#undef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0
#else
// Starting from 4.5, GCC has a suppression option
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#endif //__INTEL_COMPILER

#include <stdio.h>
#include "machine.h"
#include "types.h"
#include "util.h"
#include "imageio.h" /* error codes etc */
#include "ppm.h"

static int getint(FILE * dfile) {
  char ch[200];
  int i;
  int num;

  num=0; 
  while (num==0) {
    fscanf(dfile, "%s", ch);
      while (ch[0]=='#') {
        fgets(ch, 200, dfile);
      }
    num=sscanf(ch, "%d", &i);
  }
  return i;
}

int readppm(char * name, int * xres, int * yres, unsigned char **imgdata) {
  char data[200];  
  FILE * ifp;
  int i;
  size_t bytesread;
  int datasize;
 
  ifp=fopen(name, "r");  
  if (ifp==NULL) {
    return IMAGEBADFILE; /* couldn't open the file */
  }
  fscanf(ifp, "%s", data);
 
  if (strcmp(data, "P6")) {
     fclose(ifp);
     return IMAGEUNSUP; /* not a format we support */
  }

  *xres=getint(ifp);
  *yres=getint(ifp);
      i=getint(ifp); /* eat the maxval number */
  fread(&i, 1, 1, ifp); /* eat the newline */ 
  datasize = 3 * (*xres) * (*yres);

  *imgdata=(unsigned char *)rt_getmem(datasize); 

  bytesread=fread(*imgdata, 1, datasize, ifp);   

  fclose(ifp);

  if (bytesread != datasize) 
    return IMAGEREADERR;
  
  return IMAGENOERR;
}
