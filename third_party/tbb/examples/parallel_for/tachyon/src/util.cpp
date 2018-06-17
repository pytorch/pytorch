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
 * util.cpp - Contains all of the timing functions for various platforms.
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "util.h"
#include "light.h"
#include "global.h"
#include "ui.h"

void rt_finalize(void);

#if !defined( _WIN32 )
#include <sys/time.h>
#include <unistd.h>

void rt_sleep(int msec) {
    usleep(msec*1000);
}

#else //_WIN32

#undef OLDUNIXTIME
#undef STDTIME

void rt_sleep(int msec) {
#if !WIN8UI_EXAMPLE
    Sleep(msec);
#else
    std::chrono::milliseconds sleep_time( msec );
    std::this_thread::sleep_for( sleep_time );
#endif
}

timer gettimer(void) {
    return GetTickCount ();
}

flt timertime(timer st, timer fn) {
   double ttime, start, end;

   start = ((double) st) / ((double) 1000.00);
     end = ((double) fn) / ((double) 1000.00);
   ttime = end - start;

   return ttime;
}
#endif  /*  _WIN32  */

/* if we're on a Unix with gettimeofday() we'll use newer timers */
#if defined( STDTIME )
  struct timezone tz;

timer gettimer(void) {
  timer t;
  gettimeofday(&t, &tz);
  return t;
} 
  
flt timertime(timer st, timer fn) {
   double ttime, start, end;

   start = (st.tv_sec+1.0*st.tv_usec / 1000000.0);
     end = (fn.tv_sec+1.0*fn.tv_usec / 1000000.0);
   ttime = end - start;

   return ttime;
}  
#endif  /*  STDTIME  */



/* use the old fashioned Unix time functions */
#if defined( OLDUNIXTIME )
timer gettimer(void) {
  return time(NULL);
}

flt timertime(timer st, timer fn) {
  return difftime(fn, st);;
}
#endif  /*  OLDUNIXTIME  */



/* random other helper utility functions */
int rt_meminuse(void) {
  return rt_mem_in_use;
}  

void * rt_getmem(unsigned int bytes) {
  void * mem;

  mem=malloc( bytes );
  if (mem!=NULL) { 
    rt_mem_in_use += bytes;
  } 
  else {
    rtbomb("No more memory!!!!");
  }
  return mem;
}

unsigned int rt_freemem(void * addr) {
  unsigned int bytes;

  free(addr);

  bytes=0;
  rt_mem_in_use -= bytes; 
  return bytes;
}

void rtbomb(const char * msg) {
    rt_ui_message(MSG_ERR, msg);
    rt_ui_message(MSG_ABORT, "Rendering Aborted.");

  rt_finalize();
  exit(1);
}

void rtmesg(const char * msg) {
    rt_ui_message(MSG_0, msg);
}
