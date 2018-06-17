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

#include <cstdlib>
#include <cmath>
#include <queue>
#include "tbb/tbb_stddef.h"
#include "tbb/spin_mutex.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "tbb/blocked_range.h"
#include "../test/harness.h"
#include "tbb/concurrent_priority_queue.h"

#pragma warning(disable: 4996)

#define IMPL_STL 0
#define IMPL_CPQ 1

using namespace tbb;

//const int contention = 75; // degree contention.  100 = 0 us busy_wait, 50 = 50*contention_unit us
const double contention_unit = 0.025; // in microseconds (us)
const double throughput_window = 30; // in seconds
const int num_initial_events = 10000; // number of initial events in the queue
const int min_elapse = 20; // min contention_units to elapse between event spawns
const int max_elapse = 40; // max contention_units to elapse between event spawns
const int min_spawn = 0; // min number of events to spawn
const int max_spawn = 2; // max number of events to spawn

tbb::atomic<unsigned int> operation_count;
tbb::tick_count start;
bool done;

class event {
public:
    int timestamp;
    int elapse;
    int spawn;
};

class timestamp_compare {
public:
    bool operator()(event e1, event e2) {
        return e2.timestamp<e1.timestamp;
    }
};

spin_mutex *my_mutex;
std::priority_queue<event, std::vector<event>, timestamp_compare > *stl_cpq;
concurrent_priority_queue<event, timestamp_compare > *lfc_pq;

unsigned int one_us_iters = 429; // default value

// if user wants to calibrate to microseconds on particular machine, call this at beginning of program
// sets one_us_iters to number of iters to busy_wait for approx. 1 us
void calibrate_busy_wait() {
    const unsigned niter = 1000000;
    tbb::tick_count t0 = tbb::tick_count::now();
    for (volatile unsigned int i=0; i<niter; ++i) continue;
    tbb::tick_count t1 = tbb::tick_count::now();

    one_us_iters = (unsigned int)(niter/(t1-t0).seconds())*1e-6;
    printf("one_us_iters: %d\n", one_us_iters);
}

void busy_wait(double us)
{
    unsigned int iter = us*one_us_iters;
    for (volatile unsigned int i=0; i<iter; ++i) continue;
}


void do_push(event elem, int nThr, int impl) {
    if (impl == IMPL_STL) {
        if (nThr == 1) {
            stl_cpq->push(elem);
        }
        else {
            tbb::spin_mutex::scoped_lock myLock(*my_mutex);
            stl_cpq->push(elem);
        }
    }
    else {
        lfc_pq->push(elem);
    }
}

bool do_pop(event& elem, int nThr, int impl) {
    if (impl == IMPL_STL) {
        if (nThr == 1) {
            if (!stl_cpq->empty()) {
                elem = stl_cpq->top();
                stl_cpq->pop();
                return true;
            }
        }
        else {
            tbb::spin_mutex::scoped_lock myLock(*my_mutex);
            if (!stl_cpq->empty()) {
                elem = stl_cpq->top();
                stl_cpq->pop();
                return true;
            }
        }
    }
    else {
        if (lfc_pq->try_pop(elem)) {
            return true;
        }
    }
    return false;
}

struct TestPDESloadBody : NoAssign {
    int nThread;
    int implementation;

    TestPDESloadBody(int nThread_, int implementation_) : 
        nThread(nThread_), implementation(implementation_) {}
    
    void operator()(const int threadID) const {
        if (threadID == nThread) {
            sleep(throughput_window);
            done = true;
        }
        else {
            event e, tmp;
            unsigned int num_operations = 0;
            for (;;) {
                // pop an event
                if (do_pop(e, nThread, implementation)) {
                    num_operations++;
                    // do the event
                    busy_wait(e.elapse*contention_unit);
                    while (e.spawn > 0) {
                        tmp.spawn = ((e.spawn+1-min_spawn) % ((max_spawn-min_spawn)+1))+min_spawn;
                        tmp.timestamp = e.timestamp + e.elapse;
                        e.timestamp = tmp.timestamp;
                        e.elapse = ((e.elapse+1-min_elapse) % ((max_elapse-min_elapse)+1))+min_elapse;
                        tmp.elapse = e.elapse;
                        do_push(tmp, nThread, implementation);
                        num_operations++;
                        e.spawn--;
                        busy_wait(e.elapse*contention_unit);
                        if (done) break;
                    }
                }
                if (done) break;
            }
            operation_count += num_operations;
        }
    }
};

void preload_queue(int nThr, int impl) {
    event an_event;
    for (int i=0; i<num_initial_events; ++i) {
        an_event.timestamp = 0;
        an_event.elapse = (int)rand() % (max_elapse+1);
        an_event.spawn = (int)rand() % (max_spawn+1);
        do_push(an_event, nThr, impl);
    }
}

void TestPDESload(int nThreads) {
    REPORT("%4d", nThreads);

    operation_count = 0;
    done = false;
    stl_cpq = new std::priority_queue<event, std::vector<event>, timestamp_compare >;
    preload_queue(nThreads, IMPL_STL);
    TestPDESloadBody my_stl_test(nThreads, IMPL_STL);
    start = tbb::tick_count::now();
    NativeParallelFor(nThreads+1, my_stl_test);
    delete stl_cpq;

    REPORT(" %10d", operation_count/throughput_window);
    
    operation_count = 0;
    done = false;
    lfc_pq = new concurrent_priority_queue<event, timestamp_compare >;
    preload_queue(nThreads, IMPL_CPQ);
    TestPDESloadBody my_cpq_test(nThreads, IMPL_CPQ);
    start = tbb::tick_count::now();
    NativeParallelFor(nThreads+1, my_cpq_test);
    delete lfc_pq;

    REPORT(" %10d\n", operation_count/throughput_window);
}

int TestMain() {
    srand(42);
    if (MinThread < 1)
        MinThread = 1;
    //calibrate_busy_wait();
    cache_aligned_allocator<spin_mutex> my_mutex_allocator;
    my_mutex = (spin_mutex *)my_mutex_allocator.allocate(1);

    REPORT("#Thr ");
    REPORT("STL        ");
#ifdef LINEARIZABLE
    REPORT("CPQ_L\n"); 
#else
    REPORT("CPQ_N\n"); 
#endif
    for (int p = MinThread; p <= MaxThread; ++p) {
        TestPDESload(p);
    }

    return Harness::Done;
}
