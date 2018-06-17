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

// Testing automatic initialization of TBB task scheduler, so do not use task_scheduler_init anywhere.

#include "tbb/task.h"

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"
#include "tbb/atomic.h"

static tbb::atomic<int> g_NumTestsExecuted;

#define TEST_PROLOGUE() ++g_NumTestsExecuted

// Global data used in testing use cases with cross-thread usage of TBB objects
static tbb::task *g_Root1 = NULL,
                 *g_Root2 = NULL,
                 *g_Root3 = NULL,
                 *g_Task = NULL;

#if __TBB_TASK_GROUP_CONTEXT
static tbb::task_group_context* g_Ctx = NULL;
#endif /* __TBB_TASK_GROUP_CONTEXT */


void TestTaskSelf () {
    TEST_PROLOGUE();
    tbb::task& t = tbb::task::self();
    ASSERT( !t.parent() && t.ref_count() == 1 && !t.affinity(), "Master's default task properties changed?" );
}

void TestRootAllocation () {
    TEST_PROLOGUE();
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task::spawn_root_and_wait(r);
}

inline void ExecuteChildAndCleanup ( tbb::task &r, tbb::task &t ) {
    r.set_ref_count(2);
    r.spawn_and_wait_for_all(t);
    r.destroy(r);
}

void TestChildAllocation () {
    TEST_PROLOGUE();
    tbb::task &t = *new( g_Root1->allocate_child() ) tbb::empty_task;
    ExecuteChildAndCleanup( *g_Root1, t );
}

void TestAdditionalChildAllocation () {
    TEST_PROLOGUE();
    tbb::task &t = *new( tbb::task::allocate_additional_child_of(*g_Root2) ) tbb::empty_task;
    ExecuteChildAndCleanup( *g_Root2, t );
}

#if __TBB_TASK_GROUP_CONTEXT
void TestTaskGroupContextCreation () {
    TEST_PROLOGUE();
    tbb::task_group_context ctx;
    tbb::task &r = *new( tbb::task::allocate_root(ctx) ) tbb::empty_task;
    tbb::task::spawn_root_and_wait(r);
}

void TestRootAllocationWithContext () {
    TEST_PROLOGUE();
    tbb::task* root = new( tbb::task::allocate_root(*g_Ctx) ) tbb::empty_task;
    tbb::task::spawn_root_and_wait(*root);
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

void TestSpawn () {
    TEST_PROLOGUE();
    tbb::task::spawn(*g_Task);
}

void TestWaitForAll () {
    TEST_PROLOGUE();
    g_Root3->wait_for_all();
    tbb::task::destroy(*g_Root3);
}

typedef void (*TestFnPtr)();

const TestFnPtr TestFuncsTable[] = {
        TestTaskSelf, TestRootAllocation, TestChildAllocation, TestAdditionalChildAllocation,
#if __TBB_TASK_GROUP_CONTEXT
        TestTaskGroupContextCreation, TestRootAllocationWithContext,
#endif /* __TBB_TASK_GROUP_CONTEXT */
        TestSpawn, TestWaitForAll };

const int NumTestFuncs = sizeof(TestFuncsTable) / sizeof(TestFnPtr);

struct TestThreadBody : NoAssign, Harness::NoAfterlife {
    // Each invocation of operator() happens in a fresh thread with zero-based ID
    // id, and checks a specific auto-initialization scenario.
    void operator() ( int id ) const {
        ASSERT( id >= 0 && id < NumTestFuncs, "Test diver: NativeParallelFor is used incorrectly" );
        TestFuncsTable[id]();
    }
};


#include "../tbb/tls.h"

void UseAFewNewTlsKeys () {
    tbb::internal::tls<intptr_t> tls1, tls2, tls3, tls4;
    tls1 = tls2 = tls3 = tls4 = -1;
}

using tbb::internal::spin_wait_until_eq;

volatile bool FafStarted   = false,
              FafCanFinish = false,
              FafCompleted = false;

//! This task is supposed to be executed during termination of an auto-initialized master thread
class FireAndForgetTask : public tbb::task {
    tbb::task* execute () __TBB_override {
        // Let another master thread proceed requesting new TLS keys
        FafStarted = true;
        UseAFewNewTlsKeys();
        // Wait while another master thread dirtied its new TLS slots
        spin_wait_until_eq( FafCanFinish, true );
        FafCompleted = true;
        return NULL;
    }
public: // to make gcc 3.2.3 happy
    ~FireAndForgetTask() {
        ASSERT(FafCompleted, "FireAndForgetTask got erroneously cancelled?");
    }
};

#include "harness_barrier.h"
Harness::SpinBarrier driver_barrier(2);

struct DriverThreadBody : NoAssign, Harness::NoAfterlife {
    void operator() ( int id ) const {
        ASSERT( id < 2, "Only two test driver threads are expected" );
        // a barrier is required to ensure both threads started; otherwise the test may deadlock:
        // the first thread would execute FireAndForgetTask at shutdown and wait for FafCanFinish,
        // while the second thread wouldn't even start waiting for the loader lock hold by the first one.
        if ( id == 0 ) {
            driver_barrier.wait();
            // Prepare global data
            g_Root1 = new( tbb::task::allocate_root() ) tbb::empty_task;
            g_Root2 = new( tbb::task::allocate_root() ) tbb::empty_task;
            g_Root3 = new( tbb::task::allocate_root() ) tbb::empty_task;
            g_Task = new( g_Root3->allocate_child() ) tbb::empty_task;
            g_Root3->set_ref_count(2);
            // Run tests
            NativeParallelFor( NumTestFuncs, TestThreadBody() );
            ASSERT( g_NumTestsExecuted == NumTestFuncs, "Test driver: Wrong number of tests executed" );

            // This test checks the validity of temporarily restoring the value of
            // the last TLS slot for a given key during the termination of an
            // auto-initialized master thread (in governor::auto_terminate).
            // If anything goes wrong, generic_scheduler::cleanup_master() will assert.
            // The context for this task must be valid till the task completion.
#if __TBB_TASK_GROUP_CONTEXT
            tbb::task &r = *new( tbb::task::allocate_root(*g_Ctx) ) FireAndForgetTask;
#else
            tbb::task &r = *new( tbb::task::allocate_root() ) FireAndForgetTask;
#endif /* __TBB_TASK_GROUP_CONTEXT */
            tbb::task::spawn(r);
        }
        else {
#if __TBB_TASK_GROUP_CONTEXT
            tbb::task_group_context ctx;
            g_Ctx = &ctx;
#endif /* __TBB_TASK_GROUP_CONTEXT */
            driver_barrier.wait();
            spin_wait_until_eq( FafStarted, true );
            UseAFewNewTlsKeys();
            FafCanFinish = true;
            spin_wait_until_eq( FafCompleted, true );
        }
    }
};

int TestMain () {
    // Do not use any TBB functionality in the main thread!
    NativeParallelFor( 2, DriverThreadBody() );
    return Harness::Done;
}
