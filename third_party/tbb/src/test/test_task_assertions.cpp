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

// Test correctness of forceful TBB initialization before any dynamic initialization
// of static objects inside the library took place.
namespace tbb {
namespace internal {
    // Forward declaration of the TBB general initialization routine from task.cpp
    void DoOneTimeInitializations();
}}

struct StaticInitializationChecker {
    StaticInitializationChecker () { tbb::internal::DoOneTimeInitializations(); }
} theChecker;

//------------------------------------------------------------------------
// Test that important assertions in class task fail as expected.
//------------------------------------------------------------------------

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness_inject_scheduler.h"
#include "harness.h"
#include "harness_bad_expr.h"

#if TRY_BAD_EXPR_ENABLED
//! Task that will be abused.
tbb::task* volatile AbusedTask;

//! Number of times that AbuseOneTask
int AbuseOneTaskRan;

//! Body used to create task in thread 0 and abuse it in thread 1.
struct AbuseOneTask {
    void operator()( int ) const {
        tbb::task_scheduler_init init;
        // Thread 1 attempts to incorrectly use the task created by thread 0.
        tbb::task_list list;
        // spawn_root_and_wait over empty list should vacuously succeed.
        tbb::task::spawn_root_and_wait(list);

        // Check that spawn_root_and_wait fails on non-empty list.
        list.push_back(*AbusedTask);

        // Try abusing recycle_as_continuation
        TRY_BAD_EXPR(AbusedTask->recycle_as_continuation(), "execute" );
        TRY_BAD_EXPR(AbusedTask->recycle_as_safe_continuation(), "execute" );
        TRY_BAD_EXPR(AbusedTask->recycle_to_reexecute(), "execute" );
        ++AbuseOneTaskRan;
    }
};

//! Test various __TBB_ASSERT assertions related to class tbb::task.
void TestTaskAssertions() {
    // Catch assertion failures
    tbb::set_assertion_handler( AssertionFailureHandler );
    tbb::task_scheduler_init init;
    // Create task to be abused
    AbusedTask = new( tbb::task::allocate_root() ) tbb::empty_task;
    NativeParallelFor( 1, AbuseOneTask() );
    ASSERT( AbuseOneTaskRan==1, NULL );
    tbb::task::destroy(*AbusedTask);
    // Restore normal assertion handling
    tbb::set_assertion_handler( ReportError );
}

int TestMain () {
    TestTaskAssertions();
    return Harness::Done;
}

#else /* !TRY_BAD_EXPR_ENABLED */

int TestMain () {
    return Harness::Skipped;
}

#endif /* !TRY_BAD_EXPR_ENABLED */
