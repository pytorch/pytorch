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

// Test that __TBB_Yield works.
// On Red Hat EL4 U1, it does not work, because sched_yield is broken.

#define HARNESS_DEFAULT_MIN_THREADS 4
#define HARNESS_DEFAULT_MAX_THREADS 8

#include "tbb/tbb_machine.h"
#include "tbb/tick_count.h"
#include "harness.h"

static volatile long CyclicCounter;
static volatile bool Quit;
double SingleThreadTime;

struct RoundRobin: NoAssign {
    const int number_of_threads;
    RoundRobin( long p ) : number_of_threads(p) {}
    void operator()( long k ) const {
        tbb::tick_count t0 = tbb::tick_count::now();
        for( long i=0; i<10000; ++i ) {
            // Wait for previous thread to notify us
            for( int j=0; CyclicCounter!=k && !Quit; ++j ) {
                __TBB_Yield();
                if( j%100==0 ) {
                    tbb::tick_count t1 = tbb::tick_count::now();
                    if( (t1-t0).seconds()>=1.0*number_of_threads ) {
                        REPORT("Warning: __TBB_Yield failing to yield with %d threads (or system is heavily loaded)\n",number_of_threads);
                        Quit = true;
                        return;
                    }
                }
            }
            // Notify next thread that it can run
            CyclicCounter = (k+1)%number_of_threads;
        }
    }
};

int TestMain () {
    for( int p=MinThread; p<=MaxThread; ++p ) {
        REMARK("testing with %d threads\n", p );
        CyclicCounter = 0;
        Quit = false;
        NativeParallelFor( long(p), RoundRobin(p) );
    }
    return Harness::Done;
}

