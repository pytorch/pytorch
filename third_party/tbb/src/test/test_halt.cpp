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

#define HARNESS_DEFAULT_MIN_THREADS 4
#define HARNESS_DEFAULT_MAX_THREADS 8

#include "harness_defs.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <utility>
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/mutex.h"
#include "tbb/spin_mutex.h"
#include "tbb/queuing_mutex.h"
#include "harness.h"

using namespace std;
using namespace tbb;

///////////////////// Parallel methods ////////////////////////

// *** Serial shared by mutexes *** //
int SharedI = 1, SharedN;
template<typename M>
class SharedSerialFibBody: NoAssign {
    M &mutex;
public:
    SharedSerialFibBody( M &m ) : mutex( m ) {}
    //! main loop
    void operator()( const blocked_range<int>& /*range*/ ) const {
        for(;;) {
            typename M::scoped_lock lock( mutex );
            if(SharedI >= SharedN) break;
            volatile double sum = 7.3;
            sum *= 11.17;
            ++SharedI;
        }
    }
};

//! Root function
template<class M>
void SharedSerialFib(int n)
{
    SharedI = 1;
    SharedN = n;
    M mutex;
    parallel_for( blocked_range<int>(0,4,1), SharedSerialFibBody<M>( mutex ) );
}

/////////////////////////// Main ////////////////////////////////////////////////////

double Tsum = 0; int Tnum = 0;

typedef void (*MeasureFunc)(int);
//! Measure ticks count in loop [2..n]
void Measure(const char *name, MeasureFunc func, int n)
{
    tick_count t0;
    tick_count::interval_t T;
    REMARK("%s",name);
    t0 = tick_count::now();
    for(int number = 2; number <= n; number++)
        func(number);
    T = tick_count::now() - t0;
    double avg = Tnum? Tsum/Tnum : 1;
    if (avg == 0.0) avg = 1;
    if(avg * 100 < T.seconds()) {
        REPORT("Warning: halting detected (%g sec, av: %g)\n", T.seconds(), avg);
        ASSERT(avg * 1000 > T.seconds(), "Too long halting period");
    } else {
        Tsum += T.seconds(); Tnum++;
    }
    REMARK("\t- in %f msec\n", T.seconds()*1000);
}

int TestMain () {
    MinThread = max(2, MinThread);
    int NumbersCount = 100;
    short recycle = 100;
    do {
        for(int threads = MinThread; threads <= MaxThread; threads++) {
            task_scheduler_init scheduler_init(threads);
            REMARK("Threads number is %d\t", threads);
            Measure("Shared serial (wrapper mutex)\t", SharedSerialFib<mutex>, NumbersCount);
            //sum = Measure("Shared serial (spin_mutex)", SharedSerialFib<tbb::spin_mutex>, NumbersCount);
            //sum = Measure("Shared serial (queuing_mutex)", SharedSerialFib<tbb::queuing_mutex>, NumbersCount);
        }
    } while(--recycle);
    return Harness::Done;
}
