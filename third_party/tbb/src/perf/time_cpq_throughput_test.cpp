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

#define HARNESS_CUSTOM_MAIN 1
#define HARNESS_NO_PARSE_COMMAND_LINE 1

#include <cstdlib>
#include <cmath>
#include <queue>
#include "tbb/tbb_stddef.h"
#include "tbb/spin_mutex.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/concurrent_priority_queue.h"
#include "../test/harness.h"
#include "../examples/common/utility/utility.h"
#if _MSC_VER
#pragma warning(disable: 4996)
#endif

#define IMPL_SERIAL 0
#define IMPL_STL 1
#define IMPL_CPQ 2

using namespace tbb;

// test parameters & defaults
int impl = IMPL_CPQ; // which implementation to test
int contention = 1; // busywork between operations in us
int preload = 0; // # elements to pre-load queue with
double throughput_window = 30.0; // in seconds
int ops_per_iteration = 20; // minimum: 2 (1 push, 1 pop)
const int sample_operations = 1000; // for timing checks

// global data & types
int pushes_per_iter;
int pops_per_iter;
tbb::atomic<unsigned int> operation_count;
tbb::tick_count start;

// a non-trivial data element to use in the priority queue
const int padding_size = 15;  // change to get cache line size for test machine
class padding_type {
public:
    int p[padding_size];
    padding_type& operator=(const padding_type& other) {
        if (this != &other) {
            for (int i=0; i<padding_size; ++i) {
                p[i] = other.p[i];
            }
        }
        return *this;
    }
};

class my_data_type {
public:
    int priority;
    padding_type padding;
    my_data_type() : priority(0) {}
};

class my_less {
public:
    bool operator()(my_data_type d1, my_data_type d2) {
        return d1.priority<d2.priority;
    }
};

// arrays to get/put data from/to to generate non-trivial accesses during busywork
my_data_type *input_data;
my_data_type *output_data;
size_t arrsz;

// Serial priority queue
std::priority_queue<my_data_type, std::vector<my_data_type>, my_less > *serial_cpq;

// Coarse-locked priority queue
spin_mutex *my_mutex;
std::priority_queue<my_data_type, std::vector<my_data_type>, my_less > *stl_cpq;

// TBB concurrent_priority_queue
concurrent_priority_queue<my_data_type, my_less > *agg_cpq;

// Busy work and calibration helpers
unsigned int one_us_iters = 345; // default value

// if user wants to calibrate to microseconds on particular machine, call 
// this at beginning of program; sets one_us_iters to number of iters to 
// busy_wait for approx. 1 us
void calibrate_busy_wait() {
    const unsigned niter = 1000000;
    tbb::tick_count t0 = tbb::tick_count::now();
    for (volatile unsigned int i=0; i<niter; ++i) continue;
    tbb::tick_count t1 = tbb::tick_count::now();

    one_us_iters = (unsigned int)(niter/(t1-t0).seconds())*1e-6;
    printf("one_us_iters: %d\n", one_us_iters);
}

void busy_wait(int us)
{
    unsigned int iter = us*one_us_iters;
    for (volatile unsigned int i=0; i<iter; ++i) continue;
}

// Push to priority queue, depending on implementation
void do_push(my_data_type elem, int nThr, int impl) {
    if (impl == IMPL_SERIAL) {
        serial_cpq->push(elem);
    }
    else if (impl == IMPL_STL) {
        tbb::spin_mutex::scoped_lock myLock(*my_mutex);
        stl_cpq->push(elem);
    }
    else if (impl == IMPL_CPQ) {
        agg_cpq->push(elem);
    }
}

// Pop from priority queue, depending on implementation
my_data_type do_pop(int nThr, int impl) {
    my_data_type elem;
    if (impl == IMPL_SERIAL) {
        if (!serial_cpq->empty()) {
            elem = serial_cpq->top();
            serial_cpq->pop();
            return elem;
        }
    }
    else if (impl == IMPL_STL) {
        tbb::spin_mutex::scoped_lock myLock(*my_mutex);
        if (!stl_cpq->empty()) {
            elem = stl_cpq->top();
            stl_cpq->pop();
            return elem;
        }
    }
    else if (impl == IMPL_CPQ) {
        if (agg_cpq->try_pop(elem)) {
            return elem;
        }
    }
    return elem;
}


struct TestThroughputBody : NoAssign {
    int nThread;
    int implementation;

    TestThroughputBody(int nThread_, int implementation_) : 
        nThread(nThread_), implementation(implementation_) {}
    
    void operator()(const int threadID) const {
        tbb::tick_count now;
        size_t pos_in = threadID, pos_out = threadID;
        my_data_type elem;
        while (1) {
            for (int i=0; i<sample_operations; i+=ops_per_iteration) {
                // do pushes
                for (int j=0; j<pushes_per_iter; ++j) {
                    elem = input_data[pos_in];
                    do_push(elem, nThread, implementation);
                    busy_wait(contention);
                    pos_in += nThread;
                    if (pos_in >= arrsz) pos_in = pos_in % arrsz;
                }
                // do pops
                for (int j=0; j<pops_per_iter; ++j) {
                    output_data[pos_out] = do_pop(nThread, implementation);
                    busy_wait(contention);
                    pos_out += nThread;
                    if (pos_out >= arrsz) pos_out = pos_out % arrsz;
                }
            }
            now = tbb::tick_count::now();
            operation_count += sample_operations;
            if ((now-start).seconds() >= throughput_window) break;
        }
    }
};

void TestSerialThroughput() {
    tbb::tick_count now;

    serial_cpq = new std::priority_queue<my_data_type, std::vector<my_data_type>, my_less >;        
    for (int i=0; i<preload; ++i) do_push(input_data[i], 1, IMPL_SERIAL);

    TestThroughputBody my_serial_test(1, IMPL_SERIAL);
    start = tbb::tick_count::now();
    NativeParallelFor(1, my_serial_test);
    now = tbb::tick_count::now();
    delete serial_cpq;

    printf("SERIAL 1 %10d\n", int(operation_count/(now-start).seconds()));
}

void TestThroughputCpqOnNThreads(int nThreads) {
    tbb::tick_count now;

    if (impl == IMPL_STL) {
        stl_cpq = new std::priority_queue<my_data_type, std::vector<my_data_type>, my_less >;
        for (int i=0; i<preload; ++i) do_push(input_data[i], nThreads, IMPL_STL);

        TestThroughputBody my_stl_test(nThreads, IMPL_STL);
        start = tbb::tick_count::now();
        NativeParallelFor(nThreads, my_stl_test);
        now = tbb::tick_count::now();
        delete stl_cpq;
        
        printf("STL  %3d %10d\n", nThreads, int(operation_count/(now-start).seconds()));
    }
    else if (impl == IMPL_CPQ) {
        agg_cpq = new concurrent_priority_queue<my_data_type, my_less >;
        for (int i=0; i<preload; ++i) do_push(input_data[i], nThreads, IMPL_CPQ);

        TestThroughputBody my_cpq_test(nThreads, IMPL_CPQ);
        start = tbb::tick_count::now();
        NativeParallelFor(nThreads, my_cpq_test);
        now = tbb::tick_count::now();
        delete agg_cpq;
        
        printf("CPQ  %3d %10d\n", nThreads, int(operation_count/(now-start).seconds()));
    }
}


int main(int argc, char *argv[]) {
    utility::thread_number_range threads(tbb::task_scheduler_init::default_num_threads);
    struct select_impl{
        static bool validate(const int & impl){
            return  ((impl == IMPL_SERIAL) || (impl == IMPL_STL) || (impl == IMPL_CPQ));
        }
    };
    utility::parse_cli_arguments(argc,argv,utility::cli_argument_pack()
            .positional_arg(threads,"n-of-threads",utility::thread_number_range_desc)
            .positional_arg(contention,"contention"," busywork between operations, in us")
            .positional_arg(impl,"queue_type", "which implementation to test. One of 0(SERIAL), 1(STL), 2(CPQ) ", select_impl::validate)
            .positional_arg(preload,"preload","number of elements to pre-load queue with")
            .positional_arg(ops_per_iteration, "batch size" ,"minimum: 2 (1 push, 1 pop)")
            .positional_arg(throughput_window, "duration", "in seconds")
            );

    std::cout<< "Priority queue performance test "<<impl<<" will run with "<<contention<<"us contention "
           "using "<<threads<<" threads, "<<ops_per_iteration<<" batch size, "<<preload<<" pre-loaded elements,"
           " for "<<throughput_window<<" seconds.\n"
           <<std::flush
    ;

    srand(42);
    arrsz = 100000;
    input_data = new my_data_type[arrsz];
    output_data = new my_data_type[arrsz];
    for (size_t i=0; i<arrsz; ++i) {
       input_data[i].priority = rand()%100;
    }
    //calibrate_busy_wait();
    pushes_per_iter = ops_per_iteration/2;
    pops_per_iter = ops_per_iteration/2;
    operation_count = 0;

    // Initialize mutex for Coarse-locked priority_queue
    cache_aligned_allocator<spin_mutex> my_mutex_allocator;
    my_mutex = (spin_mutex *)my_mutex_allocator.allocate(1);

    if (impl == IMPL_SERIAL) {
        TestSerialThroughput();
    }
    else {
        for( int p=threads.first; p<=threads.last; p = threads.step(p) ) {
            TestThroughputCpqOnNThreads(p);
        }
    }
    return Harness::Done;
}
