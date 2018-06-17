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

// Declarations for simple estimate of CPU time being used by a program.
// This header is an optional part of the test harness.
// It assumes that "harness_assert.h" has already been included.

#if _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
    #include <sys/resource.h>
#endif

//! Return time (in seconds) spent by the current process in user mode.
/*  Returns 0 if not implemented on platform. */
static double GetCPUUserTime() {
#if __TBB_WIN8UI_SUPPORT
    return 0;
#elif _WIN32
    FILETIME my_times[4];
    bool status = GetProcessTimes(GetCurrentProcess(), my_times, my_times+1, my_times+2, my_times+3)!=0;
    ASSERT( status, NULL );
    LARGE_INTEGER usrtime;
    usrtime.LowPart = my_times[3].dwLowDateTime;
    usrtime.HighPart = my_times[3].dwHighDateTime;
    return double(usrtime.QuadPart)*1E-7;
#else
    // Generic UNIX, including __APPLE__

    // On Linux, there is no good way to get CPU usage info for the current process:
    //   getrusage(RUSAGE_SELF, ...) that is used now only returns info for the calling thread;
    //   getrusage(RUSAGE_CHILDREN, ...) only counts for finished children threads;
    //   tms_utime and tms_cutime got with times(struct tms*) are equivalent to the above items;
    //   finally, /proc/self/task/<task_id>/stat doesn't exist on older kernels
    //      and it isn't quite convenient to read it for every task_id.

    struct rusage resources;
    bool status = getrusage(RUSAGE_SELF, &resources)==0;
    ASSERT( status, NULL );
    return (double(resources.ru_utime.tv_sec)*1E6 + double(resources.ru_utime.tv_usec))*1E-6;
#endif
}

#include "tbb/tick_count.h"
#include <cstdio>

// The resolution of GetCPUUserTime is 10-15 ms or so; waittime should be a few times bigger.
const double WAITTIME = 0.1; // in seconds, i.e. 100 ms
const double THRESHOLD = WAITTIME/100;

static void TestCPUUserTime( int nthreads, int nactive = 1 ) {
    // The test will always pass on Linux; read the comments in GetCPUUserTime for details
    // Also it will not detect spinning issues on systems with only one processing core.

    int nworkers = nthreads-nactive;
    if( !nworkers ) return;
    double lastusrtime = GetCPUUserTime();
    if( !lastusrtime ) return;

    static double minimal_waittime = WAITTIME,
                  maximal_waittime = WAITTIME * 10;
    double usrtime_delta;
    double waittime_delta;
    tbb::tick_count stamp = tbb::tick_count::now();
    volatile intptr_t k = (intptr_t)&usrtime_delta;
    // wait for GetCPUUserTime update
    while( (usrtime_delta=GetCPUUserTime()-lastusrtime) < THRESHOLD ) {
        for ( int i = 0; i < 1000; ++i ) ++k; // do fake work without which user time can stall
        if ( (waittime_delta = (tbb::tick_count::now()-stamp).seconds()) > maximal_waittime ) {
            REPORT( "Warning: %.2f sec elapsed but user mode time is still below its threshold (%g < %g)\n",
                    waittime_delta, usrtime_delta, THRESHOLD );
            break;
        }
    }
    lastusrtime += usrtime_delta;

    // Wait for workers to go sleep
    stamp = tbb::tick_count::now();
    while( ((waittime_delta=(tbb::tick_count::now()-stamp).seconds()) < minimal_waittime)
            || ((usrtime_delta=GetCPUUserTime()-lastusrtime) < THRESHOLD) )
    {
        for ( int i = 0; i < 1000; ++i ) ++k; // do fake work without which user time can stall
        if ( waittime_delta > maximal_waittime ) {
            REPORT( "Warning: %.2f sec elapsed but GetCPUUserTime reported only %g sec\n", waittime_delta, usrtime_delta );
            break;
        }
    }

    // Test that all workers sleep when no work.
    while( nactive>1 && usrtime_delta-nactive*waittime_delta<0 ) {
        // probably the number of active threads was mispredicted
        --nactive; ++nworkers;
    }
    double avg_worker_usrtime = (usrtime_delta-nactive*waittime_delta)/nworkers;

    if( avg_worker_usrtime > waittime_delta/2 )
        REPORT( "ERROR: %d worker threads are spinning; waittime: %g; usrtime: %g; avg worker usrtime: %g\n",
                nworkers, waittime_delta, usrtime_delta, avg_worker_usrtime);
    else
        REMARK("%d worker threads; waittime: %g; usrtime: %g; avg worker usrtime: %g\n",
                        nworkers, waittime_delta, usrtime_delta, avg_worker_usrtime);
}
