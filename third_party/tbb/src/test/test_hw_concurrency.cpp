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

#include "harness_defs.h"

#if __TBB_TEST_SKIP_AFFINITY
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
int TestMain() {
    return Harness::Skipped;
}
#else /* affinity mask can be set and used by TBB */

#include "harness.h"
#include "harness_concurrency.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/tbb_thread.h"
#include "tbb/enumerable_thread_specific.h"

// The declaration of a global ETS object is needed to check that
// it does not initialize the task scheduler, and in particular
// does not set the default thread number. TODO: add other objects
// that should not initialize the scheduler.
tbb::enumerable_thread_specific<std::size_t> ets;

int TestMain () {
    int maxProcs = Harness::GetMaxProcs();

    if ( maxProcs < 2 )
        return Harness::Skipped;

    int availableProcs = maxProcs/2;
    ASSERT( Harness::LimitNumberOfThreads( availableProcs ) == availableProcs, "LimitNumberOfThreads has not set the requested limitation." );
    ASSERT( tbb::task_scheduler_init::default_num_threads() == availableProcs, NULL );
    ASSERT( (int)tbb::tbb_thread::hardware_concurrency() == availableProcs, NULL );
    return Harness::Done;
}
#endif /* __TBB_TEST_SKIP_AFFINITY */
