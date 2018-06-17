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

#include "tbb/task.h"
#include "harness.h"
#include "tbb/task_scheduler_init.h"

using tbb::task;

#if __TBB_ipf
    const unsigned StackSize = 1024*1024*6;
#else /*  */
    const unsigned StackSize = 1024*1024*3;
#endif

// GCC and ICC on Linux store TLS data in the stack space. This test makes sure
// that the stealing limiting heuristic used by the task scheduler does not
// switch off stealing when a large amount of TLS data is reserved.
#if _MSC_VER
__declspec(thread)
#elif __linux__ || ((__MINGW32__ || __MINGW64__) && __TBB_GCC_VERSION >= 40500)
__thread
#endif
    char map2[1024*1024*2];

class TestTask : public task {
public:
    static volatile int completed;
    task* execute() __TBB_override {
        completed = 1;
        return NULL;
    };
};

volatile int TestTask::completed = 0;

void TestStealingIsEnabled () {
    tbb::task_scheduler_init init(2, StackSize);
    task &r = *new( task::allocate_root() ) tbb::empty_task;
    task &t = *new( r.allocate_child() ) TestTask;
    r.set_ref_count(2);
    r.spawn(t);
    int count = 0;
    while ( !TestTask::completed && ++count < 6 )
        Harness::Sleep(1000);
    ASSERT( TestTask::completed, "Stealing is disabled or the machine is heavily oversubscribed" );
    r.wait_for_all();
    task::destroy(r);
}

int TestMain () {
#if !__TBB_THREAD_LOCAL_VARIABLES_PRESENT
    REPORT( "Known issue: Test skipped because no compiler support for __thread keyword.\n" );
    return Harness::Skipped;
#endif
    if ( tbb::task_scheduler_init::default_num_threads() == 1 ) {
        REPORT( "Known issue: Test requires at least 2 hardware threads.\n" );
        return Harness::Skipped;
    }
    TestStealingIsEnabled();
    return Harness::Done;
}
