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
 * tgafile.cpp - This file contains the code to write 24 bit targa files...
 */

#include "machine.h"
#include "types.h"
#include "util.h"
#include "ui.h"
#include "imageio.h"
#include "tgafile.h"

void createtgafile(char *name, unsigned short width, unsigned short height) {
  int filesize;
  FILE * ofp;

  filesize = 3*width*height + 18 - 10;
  
  if (name==NULL) 
    exit(1);
  else {
    ofp=fopen(name, "w+b");
    if (ofp == NULL) {
      char msgtxt[2048];
      sprintf(msgtxt, "Cannot create %s for output!", name);
      rt_ui_message(MSG_ERR, msgtxt);
      rt_ui_message(MSG_ABORT, "Rendering Aborted.");
      exit(1);
    } 

    fputc(0, ofp); /* IdLength      */
    fputc(0, ofp); /* ColorMapType  */
    fputc(2, ofp); /* ImageTypeCode */
    fputc(0, ofp); /* ColorMapOrigin, low byte */
    fputc(0, ofp); /* ColorMapOrigin, high byte */
    fputc(0, ofp); /* ColorMapLength, low byte */
    fputc(0, ofp); /* ColorMapLength, high byte */
    fputc(0, ofp); /* ColorMapEntrySize */
    fputc(0, ofp); /* XOrigin, low byte */
    fputc(0, ofp); /* XOrigin, high byte */
    fputc(0, ofp); /* YOrigin, low byte */
    fputc(0, ofp); /* YOrigin, high byte */
    fputc((width & 0xff),         ofp); /* Width, low byte */
    fputc(((width >> 8) & 0xff),  ofp); /* Width, high byte */
    fputc((height & 0xff),        ofp); /* Height, low byte */
    fputc(((height >> 8) & 0xff), ofp); /* Height, high byte */
    fputc(24, ofp);   /* ImagePixelSize */
    fputc(0x20, ofp); /* ImageDescriptorByte 0x20 == flip vertically */

    fseek(ofp, filesize, 0);
    fprintf(ofp, "9876543210"); 

    fclose(ofp);
  } 
}    

void * opentgafile(char * filename) {
  FILE * ofp;

  ofp=fopen(filename, "r+b");
  if (ofp == NULL) {
    char msgtxt[2048];
    sprintf(msgtxt, "Cannot open %s for output!", filename);
    rt_ui_message(MSG_ERR, msgtxt);
    rt_ui_message(MSG_ABORT, "Rendering Aborted.");
    exit(1);
  } 

  return ofp;
} 

void writetgaregion(void * voidofp, 
                    int iwidth, int iheight,
                    int startx, int starty, 
                    int stopx, int stopy, char * buffer) {
  int y, totalx, totaly;
  char * bufpos;
  long filepos;
  size_t numbytes;
  FILE * ofp = (FILE *) voidofp;
 
  totalx = stopx - startx + 1;
  totaly = stopy - starty + 1;

  for (y=0; y<totaly; y++) {
    bufpos=buffer + (totalx*3)*(totaly-y-1);
    filepos=18 + iwidth*3*(iheight - starty - totaly + y + 1) + (startx - 1)*3;

    if (filepos >= 18) {
      fseek(ofp, filepos, 0); 
      numbytes = fwrite(bufpos, 3, totalx, ofp);

      if (numbytes != totalx) {
        char msgtxt[256];
        sprintf(msgtxt, "File write problem, %d bytes written.", (int)numbytes);
        rt_ui_message(MSG_ERR, msgtxt);
      }
    }
    else {
      rt_ui_message(MSG_ERR, "writetgaregion: file ptr out of range!!!\n");
      return;  /* don't try to continue */
    }
  }
}


int readtga(char * name, int * xres, int * yres, unsigned char **imgdata) {
  int format, width, height, w1, w2, h1, h2, depth, flags;
  int imgsize, i, tmp;
  size_t bytesread;
  FILE * ifp;

  ifp=fopen(name, "r");  
  if (ifp==NULL) {
    return IMAGEBADFILE; /* couldn't open the file */
  }

  /* read the targa header */
  getc(ifp); /* ID length */
  getc(ifp); /* colormap type */
  format = getc(ifp); /* image type */
  getc(ifp); /* color map origin */
  getc(ifp); /* color map origin */
  getc(ifp); /* color map length */
  getc(ifp); /* color map length */
  getc(ifp); /* color map entry size */
  getc(ifp); /* x origin */
  getc(ifp); /* x origin */
  getc(ifp); /* y origin */
  getc(ifp); /* y origin */
  w1 = getc(ifp); /* width (low) */
  w2 = getc(ifp); /* width (hi) */
  h1 = getc(ifp); /* height (low) */
  h2 = getc(ifp); /* height (hi) */
  depth = getc(ifp); /* image pixel size */
  flags = getc(ifp); /* image descriptor byte */

  if ((format != 2) || (depth != 24)) {
    fclose(ifp);
    return IMAGEUNSUP; /* unsupported targa format */
  }
    

  width = ((w2 << 8) | w1);
  height = ((h2 << 8) | h1);

  imgsize = 3 * width * height;
  *imgdata = (unsigned char *)rt_getmem(imgsize);
  bytesread = fread(*imgdata, 1, imgsize, ifp);
  fclose(ifp);

  /* flip image vertically */
  if (flags == 0x20) {
    int rowsize = 3 * width;
    unsigned char * copytmp;

    copytmp = (unsigned char *)malloc(rowsize);

    for (i=0; i<height / 2; i++) {
      memcpy(copytmp, &((*imgdata)[rowsize*i]), rowsize);
      memcpy(&(*imgdata)[rowsize*i], &(*imgdata)[rowsize*(height - 1 - i)], rowsize);
      memcpy(&(*imgdata)[rowsize*(height - 1 - i)], copytmp, rowsize);
    }

    free(copytmp);       
  }


  /* convert from BGR order to RGB order */
  for (i=0; i<imgsize; i+=3) {
    tmp = (*imgdata)[i]; /* Blue */
    (*imgdata)[i] = (*imgdata)[i+2]; /* Red */
    (*imgdata)[i+2] = tmp; /* Blue */
  }

  *xres = width;
  *yres = height;

  if (bytesread != imgsize) 
    return IMAGEREADERR;

  return IMAGENOERR;
}






