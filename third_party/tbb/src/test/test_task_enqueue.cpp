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

#include "harness_task.h"
#include "harness_barrier.h"
#include "tbb/atomic.h"
#include "tbb/tbb_thread.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

////////////////////////////////////////////////////////////////////////////////
// Test for basic FIFO scheduling functionality

const int PairsPerTrack = 100;

class EnqueuedTask : public tbb::task {
    task* my_successor;
    int my_enqueue_order;
    int* my_track;
    tbb::task* execute() __TBB_override {
        // Capture execution order in the very beginning
        int execution_order = 2 - my_successor->decrement_ref_count();
        // Create some local work.
        TaskGenerator& p = *new( allocate_root() ) TaskGenerator(2,2);
        spawn_root_and_wait(p);
        if( execution_order==2 ) { // the "slower" of two peer tasks
            ++nCompletedPairs;
            // Of course execution order can differ from dequeue order.
            // But there is no better approximation at hand; and a single worker
            // will execute in dequeue order, which is enough for our check.
            if (my_enqueue_order==execution_order)
                ++nOrderedPairs;
            FireTwoTasks(my_track);
            destroy(*my_successor);
        }
        return NULL;
    }
public:
    EnqueuedTask( task* successor, int enq_order, int* track )
    : my_successor(successor), my_enqueue_order(enq_order), my_track(track) {}

    // Create and enqueue two tasks
    static void FireTwoTasks( int* track ) {
        int progress = ++*track;
        if( progress < PairsPerTrack ) {
            task* successor = new (allocate_root()) tbb::empty_task;
            successor->set_ref_count(2);
            enqueue( *new (allocate_root()) EnqueuedTask(successor, 1, track) );
            enqueue( *new (allocate_root()) EnqueuedTask(successor, 2, track) );
        }
    }

    static tbb::atomic<int> nCompletedPairs;
    static tbb::atomic<int> nOrderedPairs;
};

tbb::atomic<int> EnqueuedTask::nCompletedPairs;
tbb::atomic<int> EnqueuedTask::nOrderedPairs;

const int nTracks = 10;
static int TaskTracks[nTracks];
const int stall_threshold = 1000000; // 1 sec

void TimedYield( double pause_time ) {
    tbb::tick_count start = tbb::tick_count::now();
    while( (tbb::tick_count::now()-start).seconds() < pause_time )
        tbb::this_tbb_thread::sleep(tbb::tick_count::interval_t(pause_time));
}

class ProgressMonitor {
public:
    void operator() ( ) {
        int track_snapshot[nTracks];
        int stall_count = 0, uneven_progress_count = 0, last_progress_mask = 0;
        for(int i=0; i<nTracks; ++i)
            track_snapshot[i]=0;
        bool completed;
        do {
            // Yield repeatedly for at least 1 usec
            TimedYield( 1E-6 );
            int overall_progress = 0, progress_mask = 0;
            const int all_progressed = (1<<nTracks) - 1;
            completed = true;
            for(int i=0; i<nTracks; ++i) {
                int ti = TaskTracks[i];
                int pi = ti-track_snapshot[i];
                if( pi ) progress_mask |= 1<<i;
                overall_progress += pi;
                completed = completed && ti==PairsPerTrack;
                track_snapshot[i]=ti;
            }
            // The constants in the next asserts are subjective and may need correction.
            if( overall_progress )
                stall_count=0;
            else {
                ++stall_count;
                // no progress; consider it dead.
                ASSERT(stall_count < stall_threshold, "no progress on enqueued tasks; deadlock, or the machine is heavily oversubscribed?");
            }
            if( progress_mask==all_progressed || progress_mask^last_progress_mask ) {
                uneven_progress_count = 0;
                last_progress_mask = progress_mask;
            }
            else if ( overall_progress > 2 ) {
                ++uneven_progress_count;
                // The threshold of 32 is 4x bigger than what was observed on a 8-core machine with oversubscription.
                ASSERT_WARNING(uneven_progress_count < 32,
                    "some enqueued tasks seem stalling; no simultaneous progress, or the machine is oversubscribed? Investigate if repeated");
            }
        } while( !completed );
    }
};

void TestEnqueue( int p ) {
    REMARK("Testing task::enqueue for %d threads\n", p);
    for(int mode=0;mode<3;++mode) {
        tbb::task_scheduler_init init(p);
        EnqueuedTask::nCompletedPairs = EnqueuedTask::nOrderedPairs = 0;
        for(int i=0; i<nTracks; ++i) {
            TaskTracks[i] = -1; // to accommodate for the starting call
            EnqueuedTask::FireTwoTasks(TaskTracks+i);
        }
        ProgressMonitor pm;
        tbb::tbb_thread thr( pm );
        if(mode==1) {
            // do some parallel work in the meantime
            for(int i=0; i<10; i++) {
                TaskGenerator& g = *new( tbb::task::allocate_root() ) TaskGenerator(2,5);
                tbb::task::spawn_root_and_wait(g);
                TimedYield( 1E-6 );
            }
        }
        if( mode==2 ) {
            // Additionally enqueue a bunch of empty tasks. The goal is to test that tasks
            // allocated and enqueued by a thread are safe to use after the thread leaves TBB.
            tbb::task* root = new (tbb::task::allocate_root()) tbb::empty_task;
            root->set_ref_count(100);
            for( int i=0; i<100; ++i )
                tbb::task::enqueue( *new (root->allocate_child()) tbb::empty_task );
            init.terminate(); // master thread deregistered
        }
        thr.join();
        ASSERT(EnqueuedTask::nCompletedPairs==nTracks*PairsPerTrack, NULL);
        ASSERT(EnqueuedTask::nOrderedPairs<EnqueuedTask::nCompletedPairs,
            "all task pairs executed in enqueue order; de facto guarantee is too strong?");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Tests for Fire-And-Forget scheduling functionality

int NumRepeats = 200;
const int MaxNumThreads = 16;
static volatile bool Finished[MaxNumThreads] = {};

static volatile bool CanStart;

//! Custom user task interface
class ITask {
public:
    virtual ~ITask() {}
    virtual void Execute() = 0;
    virtual void Release() { delete this; }
};

class TestTask : public ITask {
    volatile bool *m_pDone;
public:
    TestTask ( volatile bool *pDone ) : m_pDone(pDone) {}

    void Execute() __TBB_override {
        *m_pDone = true;
    }
};

class CarrierTask : public tbb::task {
    ITask* m_pTask;
public:
    CarrierTask(ITask* pTask) : m_pTask(pTask) {}

    task* execute() __TBB_override {
        m_pTask->Execute();
        m_pTask->Release();
        return NULL;
    }
};

class SpawnerTask : public ITask {
    ITask* m_taskToSpawn;
public:
    SpawnerTask(ITask* job) : m_taskToSpawn(job) {}

    void Execute() __TBB_override {
        while ( !CanStart )
            __TBB_Yield();
        Harness::Sleep(10); // increases probability of the bug
        tbb::task::enqueue( *new( tbb::task::allocate_root() ) CarrierTask(m_taskToSpawn) );
    }
};

class EnqueuerBody {
public:
    void operator() ( int id ) const {
        tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads() + 1);

        SpawnerTask* pTask = new SpawnerTask( new TestTask(Finished + id) );
        tbb::task::enqueue( *new( tbb::task::allocate_root() ) CarrierTask(pTask) );
    }
};

//! Regression test for a bug that caused premature arena destruction
void TestCascadedEnqueue () {
    REMARK("Testing cascaded enqueue\n");
    tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads() + 1);

    int minNumThreads = min(tbb::task_scheduler_init::default_num_threads(), MaxNumThreads) / 2;
    int maxNumThreads = min(tbb::task_scheduler_init::default_num_threads() * 2, MaxNumThreads);

    for ( int numThreads = minNumThreads; numThreads <= maxNumThreads; ++numThreads ) {
        for ( int i = 0; i < NumRepeats; ++i ) {
            CanStart = false;
            __TBB_Yield();
            NativeParallelFor( numThreads, EnqueuerBody() );
            CanStart = true;
            int j = 0;
            while ( j < numThreads ) {
                if ( Finished[j] )
                    ++j;
                else
                    __TBB_Yield();
            }
            for ( j = 0; j < numThreads; ++j )
                Finished[j] = false;
            REMARK("\r%02d threads; Iteration %03d", numThreads, i);
        }
    }
    REMARK( "\r                                 \r" );
}

class DummyTask : public tbb::task {
public:
    task *execute() __TBB_override {
        Harness::Sleep(1);
        return NULL;
    }
};

class SharedRootBody {
    tbb::task *my_root;
public:
    SharedRootBody ( tbb::task *root ) : my_root(root) {}

    void operator() ( int ) const {
        tbb::task::enqueue( *new( tbb::task::allocate_additional_child_of(*my_root) ) DummyTask );
    }
};

//! Test for enqueuing children of the same root from different master threads
void TestSharedRoot ( int p ) {
    REMARK("Testing enqueuing siblings from different masters\n");
    tbb::task_scheduler_init init(p);
    tbb::task *root =  new ( tbb::task::allocate_root() ) tbb::empty_task;
    root->set_ref_count(1);
    for( int n = MinThread; n <= MaxThread; ++n ) {
        REMARK("%d masters, %d requested workers\r", n, p-1);
        NativeParallelFor( n, SharedRootBody(root) );
    }
    REMARK( "                                    \r" );
    root->wait_for_all();
    tbb::task::destroy(*root);
}

class BlockingTask : public tbb::task {
    Harness::SpinBarrier &m_Barrier;

    tbb::task* execute () __TBB_override {
        m_Barrier.wait();
        return 0;
    }

public:
    BlockingTask ( Harness::SpinBarrier& bar ) : m_Barrier(bar) {}
};

//! Test making sure that masters can dequeue tasks
/** Success criterion is not hanging. **/
void TestDequeueByMaster () {
    REMARK("Testing task dequeuing by master\n");
    tbb::task_scheduler_init init(1);
    Harness::SpinBarrier bar(2);
    tbb::task &r = *new ( tbb::task::allocate_root() ) tbb::empty_task;
    r.set_ref_count(3);
    tbb::task::enqueue( *new(r.allocate_child()) BlockingTask(bar) );
    tbb::task::enqueue( *new(r.allocate_child()) BlockingTask(bar) );
    r.wait_for_all();
    tbb::task::destroy(r);
}

////////////////////// Missed wake-ups ///////
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

static const int NUM_TASKS    = 4;
static const size_t NUM_REPEATS = TBB_USE_DEBUG ? 50000 : 100000;
static tbb::task_group_context persistent_context(tbb::task_group_context::isolated);

struct Functor : NoAssign
{
    Harness::SpinBarrier &my_barrier;
    Functor(Harness::SpinBarrier &a_barrier) : my_barrier(a_barrier) { }
    void operator()(const tbb::blocked_range<int>& r) const
    {
        ASSERT(r.size() == 1, NULL);
        // allocate_root() uses current context of parallel_for which is destroyed when it finishes.
        // But enqueued tasks can outlive parallel_for execution. Thus, use a persistent context.
        tbb::task *t = new(tbb::task::allocate_root(persistent_context)) tbb::empty_task();
        tbb::task::enqueue(*t); // ensure no missing wake-ups
        my_barrier.timed_wait(10, "Attention: poorly reproducible event, if seen stress testing required" );
    }
};

void TestWakeups()
{
    tbb::task_scheduler_init my(tbb::task_scheduler_init::deferred);
    if( tbb::task_scheduler_init::default_num_threads() <= NUM_TASKS )
        my.initialize(NUM_TASKS*2);
    else // workaround issue #1996 for TestCascadedEnqueue
        my.initialize(tbb::task_scheduler_init::default_num_threads()+1);
    Harness::SpinBarrier barrier(NUM_TASKS);
    REMARK("Missing wake-up: affinity_partitioner\n");
    tbb::affinity_partitioner aff;
    for (size_t i = 0; i < NUM_REPEATS; ++i)
        tbb::parallel_for(tbb::blocked_range<int>(0, NUM_TASKS), Functor(barrier), aff);
    REMARK("Missing wake-up: simple_partitioner\n");
    for (size_t i = 0; i < NUM_REPEATS; ++i)
        tbb::parallel_for(tbb::blocked_range<int>(0, NUM_TASKS), Functor(barrier), tbb::simple_partitioner());
    REMARK("Missing wake-up: auto_partitioner\n");
    for (size_t i = 0; i < NUM_REPEATS; ++i)
        tbb::parallel_for(tbb::blocked_range<int>(0, NUM_TASKS), Functor(barrier)); // auto
}

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"

int TestMain () {

    TestWakeups();         // 1st because requests oversubscription
    for (int i=0; i<2; i++) {
        tbb::global_control *c = i?
            new tbb::global_control(tbb::global_control::max_allowed_parallelism, 1) : NULL;
        if (i) // decrease workload for max_allowed_parallelism == 1
            NumRepeats = 10;

        TestCascadedEnqueue(); // needs oversubscription
        if (!c)
            TestDequeueByMaster(); // no oversubscription needed
        for( int p=MinThread; p<=MaxThread; ++p ) {
            TestEnqueue(p);
            TestSharedRoot(p);
        }
        delete c;
    }
    return Harness::Done;
}
