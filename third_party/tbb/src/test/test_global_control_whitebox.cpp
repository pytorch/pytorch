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

#define HARNESS_DEFINE_PRIVATE_PUBLIC 1
#include "harness_inject_scheduler.h"
#include "harness.h"

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

bool allWorkersSleep() {
    using namespace tbb::internal;
    using namespace tbb::internal::rml;

    unsigned sleeping_threads = 0;
    unsigned threads = ((private_server*)market::theMarket->my_server)->my_n_thread;

    for (private_worker *l = ((private_server*)market::theMarket->my_server)->my_asleep_list_root;
         l; l = l->my_next)
        sleeping_threads++;

    return threads == sleeping_threads;
}

class ThreadsTask {
public:
    void operator() (const tbb::blocked_range<int> &) const { }
    ThreadsTask() {}
};

static void RunAndCheckSleeping()
{
    Harness::Sleep(100);
    ASSERT(allWorkersSleep(), NULL);
    tbb::parallel_for(tbb::blocked_range<int>(0, 100*1000, 1),
                      ThreadsTask(), tbb::simple_partitioner());
    Harness::Sleep(100);
    ASSERT(allWorkersSleep(), NULL);
}

// test that all workers are sleeping, not spinning
void TestWorkersSleep() {
    tbb::task_scheduler_init tsi(8);
    const size_t max_parallelism =
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    if (max_parallelism > 2) {
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, max_parallelism-1);
    }
    RunAndCheckSleeping();
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, max_parallelism+1);
    RunAndCheckSleeping();
}

int TestMain () {
    {
        tbb::task_scheduler_init tsi;
        if (!tbb::internal::governor::UsePrivateRML)
            return Harness::Skipped;
    }
    TestWorkersSleep();

    return Harness::Done;
}
