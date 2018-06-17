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
#include "tbb/atomic.h"
#include "tbb/tbb_thread.h"
#include "tbb/task_scheduler_init.h"
#include <cstdlib>

//------------------------------------------------------------------------
// Test for task::spawn_children and task_list
//------------------------------------------------------------------------

class UnboundedlyRecursiveOnUnboundedStealingTask : public tbb::task {
    typedef UnboundedlyRecursiveOnUnboundedStealingTask this_type;

    this_type *m_Parent;
    const int m_Depth;
    volatile bool m_GoAhead;

    // Well, virtually unboundedly, for any practical purpose
    static const int max_depth = 1000000;

public:
    UnboundedlyRecursiveOnUnboundedStealingTask( this_type *parent_ = NULL, int depth_ = max_depth )
        : m_Parent(parent_)
        , m_Depth(depth_)
        , m_GoAhead(true)
    {}

    tbb::task* execute() __TBB_override {
        // Using large padding array speeds up reaching stealing limit
        const int paddingSize = 16 * 1024;
        volatile char padding[paddingSize];
        if( !m_Parent || (m_Depth > 0 &&  m_Parent->m_GoAhead) ) {
            if ( m_Parent ) {
                // We are stolen, let our parent start waiting for us
                m_Parent->m_GoAhead = false;
            }
            tbb::task &t = *new( allocate_child() ) this_type(this, m_Depth - 1);
            set_ref_count( 2 );
            spawn( t );
            // Give a willing thief a chance to steal
            for( int i = 0; i < 1000000 && m_GoAhead; ++i ) {
                ++padding[i % paddingSize];
                __TBB_Yield();
            }
            // If our child has not been stolen yet, then prohibit it siring ones
            // of its own (when this thread executes it inside the next wait_for_all)
            m_GoAhead = false;
            wait_for_all();
        }
        return NULL;
    }
}; // UnboundedlyRecursiveOnUnboundedStealingTask

tbb::atomic<int> Count;

class RecursiveTask: public tbb::task {
    const int m_ChildCount;
    const int m_Depth;
    //! Spawn tasks in list.  Exact method depends upon m_Depth&bit_mask.
    void SpawnList( tbb::task_list& list, int bit_mask ) {
        if( m_Depth&bit_mask ) {
            // Take address to check that signature of spawn(task_list&) is static.
            void (*s)(tbb::task_list&) = &tbb::task::spawn;
            (*s)(list);
            ASSERT( list.empty(), NULL );
            wait_for_all();
        } else {
            spawn_and_wait_for_all(list);
            ASSERT( list.empty(), NULL );
        }
    }
public:
    RecursiveTask( int child_count, int depth_ ) : m_ChildCount(child_count), m_Depth(depth_) {}
    tbb::task* execute() __TBB_override {
        ++Count;
        if( m_Depth>0 ) {
            tbb::task_list list;
            ASSERT( list.empty(), NULL );
            for( int k=0; k<m_ChildCount; ++k ) {
                list.push_back( *new( allocate_child() ) RecursiveTask(m_ChildCount/2,m_Depth-1 ) );
                ASSERT( !list.empty(), NULL );
            }
            set_ref_count( m_ChildCount+1 );
            SpawnList( list, 1 );
            // Now try reusing this as the parent.
            set_ref_count(2);
            list.push_back( *new ( allocate_child() ) tbb::empty_task() );
            SpawnList( list, 2 );
        }
        return NULL;
    }
};

//! Compute what Count should be after RecursiveTask(child_count,depth) runs.
static int Expected( int child_count, int depth ) {
    return depth<=0 ? 1 : 1+child_count*Expected(child_count/2,depth-1);
}

void TestStealLimit( int nthread ) {
#if __TBB_DEFINE_MIC
    REMARK( "skipping steal limiting heuristics for %d threads\n", nthread );
#else// !_TBB_DEFINE_MIC
    REMARK( "testing steal limiting heuristics for %d threads\n", nthread );
    tbb::task_scheduler_init init(nthread);
    tbb::task &t = *new( tbb::task::allocate_root() ) UnboundedlyRecursiveOnUnboundedStealingTask();
    tbb::task::spawn_root_and_wait(t);
#endif// _TBB_DEFINE_MIC
}

//! Test task::spawn( task_list& )
void TestSpawnChildren( int nthread ) {
    REMARK("testing task::spawn(task_list&) for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=0; j<50; ++j ) {
        Count = 0;
        RecursiveTask& p = *new( tbb::task::allocate_root() ) RecursiveTask(j,4);
        tbb::task::spawn_root_and_wait(p);
        int expected = Expected(j,4);
        ASSERT( Count==expected, NULL );
    }
}

//! Test task::spawn_root_and_wait( task_list& )
void TestSpawnRootList( int nthread ) {
    REMARK("testing task::spawn_root_and_wait(task_list&) for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=0; j<5; ++j )
        for( int k=0; k<10; ++k ) {
            Count = 0;
            tbb::task_list list;
            for( int i=0; i<k; ++i )
                list.push_back( *new( tbb::task::allocate_root() ) RecursiveTask(j,4) );
            tbb::task::spawn_root_and_wait(list);
            int expected = k*Expected(j,4);
            ASSERT( Count==expected, NULL );
        }
}

//------------------------------------------------------------------------
// Test for task::recycle_as_safe_continuation
//------------------------------------------------------------------------

void TestSafeContinuation( int nthread ) {
    REMARK("testing task::recycle_as_safe_continuation for %d threads\n",nthread);
    tbb::task_scheduler_init init(nthread);
    for( int j=8; j<33; ++j ) {
        TaskGenerator& p = *new( tbb::task::allocate_root() ) TaskGenerator(j,5);
        tbb::task::spawn_root_and_wait(p);
    }
}

//------------------------------------------------------------------------
// Test affinity interface
//------------------------------------------------------------------------
tbb::atomic<int> TotalCount;

struct AffinityTask: public tbb::task {
    const affinity_id expected_affinity_id;
    bool noted;
    /** Computing affinities is NOT supported by TBB, and may disappear in the future.
        It is done here for sake of unit testing. */
    AffinityTask( int expected_affinity_id_ ) :
        expected_affinity_id(affinity_id(expected_affinity_id_)),
        noted(false)
    {
        set_affinity(expected_affinity_id);
        ASSERT( 0u-expected_affinity_id>0u, "affinity_id not an unsigned integral type?" );
        ASSERT( affinity()==expected_affinity_id, NULL );
    }
    tbb::task* execute() __TBB_override {
        ++TotalCount;
        return NULL;
    }
    void note_affinity( affinity_id id ) __TBB_override {
        // There is no guarantee in TBB that a task runs on its affinity thread.
        // However, the current implementation does accidentally guarantee it
        // under certain conditions, such as the conditions here.
        // We exploit those conditions for sake of unit testing.
        ASSERT( id!=expected_affinity_id, NULL );
        ASSERT( !noted, "note_affinity_id called twice!" );
        ASSERT ( &self() == (tbb::task*)this, "Wrong innermost running task" );
        noted = true;
    }
};

/** Note: This test assumes a lot about the internal implementation of affinity.
    Do NOT use this as an example of good programming practice with TBB */
void TestAffinity( int nthread ) {
    TotalCount = 0;
    int n = tbb::task_scheduler_init::default_num_threads();
    if( n>nthread )
        n = nthread;
    tbb::task_scheduler_init init(n);
    tbb::empty_task* t = new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task::affinity_id affinity_id = t->affinity();
    ASSERT( affinity_id==0, NULL );
    // Set ref_count for n-1 children, plus 1 for the wait.
    t->set_ref_count(n);
    // Spawn n-1 affinitized children.
    for( int i=1; i<n; ++i )
        tbb::task::spawn( *new(t->allocate_child()) AffinityTask(i) );
    if( n>1 ) {
        // Keep master from stealing
        while( TotalCount!=n-1 )
            __TBB_Yield();
    }
    // Wait for the children
    t->wait_for_all();
    int k = 0;
    GetTaskPtr(k)->destroy(*t);
    ASSERT(k==1,NULL);
}

struct NoteAffinityTask: public tbb::task {
    bool noted;
    NoteAffinityTask( int id ) : noted(false)
    {
        set_affinity(affinity_id(id));
    }
    ~NoteAffinityTask () {
        ASSERT (noted, "note_affinity has not been called");
    }
    tbb::task* execute() __TBB_override {
        return NULL;
    }
    void note_affinity( affinity_id /*id*/ ) __TBB_override {
        noted = true;
        ASSERT ( &self() == (tbb::task*)this, "Wrong innermost running task" );
    }
};

// This test checks one of the paths inside the scheduler by affinitizing the child task
// to non-existent thread so that it is proxied in the local task pool but not retrieved
// by another thread.
// If no workers requested, the extra slot #2 is allocated for a worker thread to serve
// "enqueued" tasks. In this test, it is used only for the affinity purpose.
void TestNoteAffinityContext() {
    tbb::task_scheduler_init init(1);
    tbb::empty_task* t = new( tbb::task::allocate_root() ) tbb::empty_task;
    t->set_ref_count(2);
    // This master in the absence of workers will have an affinity id of 1.
    // So use another number to make the task get proxied.
    tbb::task::spawn( *new(t->allocate_child()) NoteAffinityTask(2) );
    t->wait_for_all();
    tbb::task::destroy(*t);
}

//------------------------------------------------------------------------
// Test that recovery actions work correctly for task::allocate_* methods
// when a task's constructor throws an exception.
//------------------------------------------------------------------------

#if TBB_USE_EXCEPTIONS
static int TestUnconstructibleTaskCount;

struct ConstructionFailure {
};

#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    // Suppress pointless "unreachable code" warning.
    #pragma warning (push)
    #pragma warning (disable: 4702)
#endif

//! Task that cannot be constructed.
template<size_t N>
struct UnconstructibleTask: public tbb::empty_task {
    char space[N];
    UnconstructibleTask() {
        throw ConstructionFailure();
    }
};

#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    #pragma warning (pop)
#endif

#define TRY_BAD_CONSTRUCTION(x)                  \
    {                                            \
        try {                                    \
            new(x) UnconstructibleTask<N>;       \
        } catch( const ConstructionFailure& ) {                                                    \
            ASSERT( parent()==original_parent, NULL ); \
            ASSERT( ref_count()==original_ref_count, "incorrectly changed ref_count" );\
            ++TestUnconstructibleTaskCount;      \
        }                                        \
    }

template<size_t N>
struct RootTaskForTestUnconstructibleTask: public tbb::task {
    tbb::task* execute() __TBB_override {
        tbb::task* original_parent = parent();
        ASSERT( original_parent!=NULL, NULL );
        int original_ref_count = ref_count();
        TRY_BAD_CONSTRUCTION( allocate_root() );
        TRY_BAD_CONSTRUCTION( allocate_child() );
        TRY_BAD_CONSTRUCTION( allocate_continuation() );
        TRY_BAD_CONSTRUCTION( allocate_additional_child_of(*this) );
        return NULL;
    }
};

template<size_t N>
void TestUnconstructibleTask() {
    TestUnconstructibleTaskCount = 0;
    tbb::task_scheduler_init init;
    tbb::task* t = new( tbb::task::allocate_root() ) RootTaskForTestUnconstructibleTask<N>;
    tbb::task::spawn_root_and_wait(*t);
    ASSERT( TestUnconstructibleTaskCount==4, NULL );
}
#endif /* TBB_USE_EXCEPTIONS */

//------------------------------------------------------------------------
// Test for alignment problems with task objects.
//------------------------------------------------------------------------

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for pointless warning "structure was padded due to __declspec(align())
    #pragma warning (push)
    #pragma warning (disable: 4324)
#endif

//! Task with members of type T.
/** The task recursively creates tasks. */
template<typename T>
class TaskWithMember: public tbb::task {
    T x;
    T y;
    unsigned char count;
    tbb::task* execute() __TBB_override {
        x = y;
        if( count>0 ) {
            set_ref_count(2);
            tbb::task* t = new( allocate_child() ) TaskWithMember<T>(count-1);
            spawn_and_wait_for_all(*t);
        }
        return NULL;
    }
public:
    TaskWithMember( unsigned char n ) : count(n) {}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif

template<typename T>
void TestAlignmentOfOneClass() {
    typedef TaskWithMember<T> task_type;
    tbb::task* t = new( tbb::task::allocate_root() ) task_type(10);
    tbb::task::spawn_root_and_wait(*t);
}

#include "harness_m128.h"

void TestAlignment() {
    REMARK("testing alignment\n");
    tbb::task_scheduler_init init;
    // Try types that have variety of alignments
    TestAlignmentOfOneClass<char>();
    TestAlignmentOfOneClass<short>();
    TestAlignmentOfOneClass<int>();
    TestAlignmentOfOneClass<long>();
    TestAlignmentOfOneClass<void*>();
    TestAlignmentOfOneClass<float>();
    TestAlignmentOfOneClass<double>();
#if HAVE_m128
    TestAlignmentOfOneClass<__m128>();
#endif
#if HAVE_m256
    if (have_AVX()) TestAlignmentOfOneClass<__m256>();
#endif
}

//------------------------------------------------------------------------
// Test for recursing on left while spawning on right
//------------------------------------------------------------------------

int Fib( int n );

struct RightFibTask: public tbb::task {
    int* y;
    const int n;
    RightFibTask( int* y_, int n_ ) : y(y_), n(n_) {}
    task* execute() __TBB_override {
        *y = Fib(n-1);
        return 0;
    }
};

int Fib( int n ) {
    if( n<2 ) {
        return n;
    } else {
        // y actually does not need to be initialized.  It is initialized solely to suppress
        // a gratuitous warning "potentially uninitialized local variable".
        int y=-1;
        tbb::task* root_task = new( tbb::task::allocate_root() ) tbb::empty_task;
        root_task->set_ref_count(2);
        tbb::task::spawn( *new( root_task->allocate_child() ) RightFibTask(&y,n) );
        int x = Fib(n-2);
        root_task->wait_for_all();
        tbb::task::destroy(*root_task);
        return y+x;
    }
}

void TestLeftRecursion( int p ) {
    REMARK("testing non-spawned roots for %d threads\n",p);
    tbb::task_scheduler_init init(p);
    int sum = 0;
    for( int i=0; i<100; ++i )
        sum +=Fib(10);
    ASSERT( sum==5500, NULL );
}

//------------------------------------------------------------------------
// Test for computing with DAG of tasks.
//------------------------------------------------------------------------

class DagTask: public tbb::task {
    typedef unsigned long long number_t;
    const int i, j;
    number_t sum_from_left, sum_from_above;
    void check_sum( number_t sum ) {
        number_t expected_sum = 1;
        for( int k=i+1; k<=i+j; ++k )
            expected_sum *= k;
        for( int k=1; k<=j; ++k )
            expected_sum /= k;
        ASSERT(sum==expected_sum, NULL);
    }
public:
    DagTask *successor_to_below, *successor_to_right;
    DagTask( int i_, int j_ ) : i(i_), j(j_), sum_from_left(0), sum_from_above(0) {}
    task* execute() __TBB_override {
        ASSERT( ref_count()==0, NULL );
        number_t sum = i==0 && j==0 ? 1 : sum_from_left+sum_from_above;
        check_sum(sum);
        ++execution_count;
        if( DagTask* t = successor_to_right ) {
            t->sum_from_left = sum;
            if( t->decrement_ref_count()==0 )
                // Test using spawn to evaluate DAG
                spawn( *t );
        }
        if( DagTask* t = successor_to_below ) {
            t->sum_from_above = sum;
            if( t->add_ref_count(-1)==0 )
                // Test using bypass to evaluate DAG
                return t;
        }
        return NULL;
    }
    ~DagTask() {++destruction_count;}
    static tbb::atomic<int> execution_count;
    static tbb::atomic<int> destruction_count;
};

tbb::atomic<int> DagTask::execution_count;
tbb::atomic<int> DagTask::destruction_count;

void TestDag( int p ) {
    REMARK("testing evaluation of DAG for %d threads\n",p);
    tbb::task_scheduler_init init(p);
    DagTask::execution_count=0;
    DagTask::destruction_count=0;
    const int n = 10;
    DagTask* a[n][n];
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j )
            a[i][j] = new( tbb::task::allocate_root() ) DagTask(i,j);
    for( int i=0; i<n; ++i )
        for( int j=0; j<n; ++j ) {
            a[i][j]->successor_to_below = i+1<n ? a[i+1][j] : NULL;
            a[i][j]->successor_to_right = j+1<n ? a[i][j+1] : NULL;
            a[i][j]->set_ref_count((i>0)+(j>0));
        }
    a[n-1][n-1]->increment_ref_count();
    a[n-1][n-1]->spawn_and_wait_for_all(*a[0][0]);
    ASSERT( DagTask::execution_count == n*n - 1, NULL );
    tbb::task::destroy(*a[n-1][n-1]);
    ASSERT( DagTask::destruction_count > n*n - p, NULL );
    while ( DagTask::destruction_count != n*n )
        __TBB_Yield();
}

#include "harness_barrier.h"

class RelaxedOwnershipTask: public tbb::task {
    tbb::task &m_taskToSpawn,
              &m_taskToDestroy,
              &m_taskToExecute;
    static Harness::SpinBarrier m_barrier;

    tbb::task* execute () __TBB_override {
        tbb::task &p = *parent();
        tbb::task &r = *new( allocate_root() ) tbb::empty_task;
        r.set_ref_count( 1 );
        m_barrier.wait();
        p.spawn( *new(p.allocate_child()) tbb::empty_task );
        p.spawn( *new(task::allocate_additional_child_of(p)) tbb::empty_task );
        p.spawn( m_taskToSpawn );
        p.destroy( m_taskToDestroy );
        r.spawn_and_wait_for_all( m_taskToExecute );
        p.destroy( r );
        return NULL;
    }
public:
    RelaxedOwnershipTask ( tbb::task& toSpawn, tbb::task& toDestroy, tbb::task& toExecute )
        : m_taskToSpawn(toSpawn)
        , m_taskToDestroy(toDestroy)
        , m_taskToExecute(toExecute)
    {}
    static void SetBarrier ( int numThreads ) { m_barrier.initialize( numThreads ); }
};

Harness::SpinBarrier RelaxedOwnershipTask::m_barrier;

void TestRelaxedOwnership( int p ) {
    if ( p < 2 )
        return;

    if( unsigned(p)>tbb::tbb_thread::hardware_concurrency() )
        return;

    REMARK("testing tasks exercising relaxed ownership freedom for %d threads\n", p);
    tbb::task_scheduler_init init(p);
    RelaxedOwnershipTask::SetBarrier(p);
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task_list tl;
    for ( int i = 0; i < p; ++i ) {
        tbb::task &tS = *new( r.allocate_child() ) tbb::empty_task,
                  &tD = *new( r.allocate_child() ) tbb::empty_task,
                  &tE = *new( r.allocate_child() ) tbb::empty_task;
        tl.push_back( *new( r.allocate_child() ) RelaxedOwnershipTask(tS, tD, tE) );
    }
    r.set_ref_count( 5 * p + 1 );
    int k=0;
    GetTaskPtr(k)->spawn( tl );
    ASSERT(k==1,NULL);
    r.wait_for_all();
    r.destroy( r );
}

//------------------------------------------------------------------------
// Test for running TBB scheduler on user-created thread.
//------------------------------------------------------------------------

void RunSchedulerInstanceOnUserThread( int n_child ) {
    tbb::task* e = new( tbb::task::allocate_root() ) tbb::empty_task;
    e->set_ref_count(1+n_child);
    for( int i=0; i<n_child; ++i )
        tbb::task::spawn( *new(e->allocate_child()) tbb::empty_task );
    e->wait_for_all();
    e->destroy(*e);
}

void TestUserThread( int p ) {
    tbb::task_scheduler_init init(p);
    // Try with both 0 and 1 children.  Only the latter scenario permits stealing.
    for( int n_child=0; n_child<2; ++n_child ) {
        tbb::tbb_thread t( RunSchedulerInstanceOnUserThread, n_child );
        t.join();
    }
}

class TaskWithChildToSteal : public tbb::task {
    const int m_Depth;
    volatile bool m_GoAhead;

public:
    TaskWithChildToSteal( int depth_ )
        : m_Depth(depth_)
        , m_GoAhead(false)
    {}

    tbb::task* execute() __TBB_override {
        m_GoAhead = true;
        if ( m_Depth > 0 ) {
            TaskWithChildToSteal &t = *new( allocate_child() ) TaskWithChildToSteal(m_Depth - 1);
            t.SpawnAndWaitOnParent();
        }
        else
            Harness::Sleep(50); // The last task in chain sleeps for 50 ms
        return NULL;
    }

    void SpawnAndWaitOnParent() {
        parent()->set_ref_count( 2 );
        parent()->spawn( *this );
        while (!this->m_GoAhead )
            __TBB_Yield();
        parent()->wait_for_all();
    }
}; // TaskWithChildToSteal

// Success criterion of this test is not hanging
void TestDispatchLoopResponsiveness() {
    REMARK("testing that dispatch loops do not go into eternal sleep when all remaining children are stolen\n");
    // Recursion depth values test the following sorts of dispatch loops
    // 0 - master's outermost
    // 1 - worker's nested
    // 2 - master's nested
    tbb::task_scheduler_init init(2);
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    for ( int depth = 0; depth < 3; ++depth ) {
        TaskWithChildToSteal &t = *new( r.allocate_child() ) TaskWithChildToSteal(depth);
        t.SpawnAndWaitOnParent();
    }
    r.destroy(r);
}

void TestWaitDiscriminativenessWithoutStealing() {
    REMARK( "testing that task::wait_for_all is specific to the root it is called on (no workers)\n" );
    // The test relies on the strict LIFO scheduling order in the absence of workers
    tbb::task_scheduler_init init(1);
    tbb::task &r1 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task &r2 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    const int NumChildren = 10;
    r1.set_ref_count( NumChildren + 1 );
    r2.set_ref_count( NumChildren + 1 );
    for( int i=0; i < NumChildren; ++i ) {
        tbb::empty_task &t1 = *new( r1.allocate_child() ) tbb::empty_task;
        tbb::empty_task &t2 = *new( r2.allocate_child() ) tbb::empty_task;
        tbb::task::spawn(t1);
        tbb::task::spawn(t2);
    }
    r2.wait_for_all();
    ASSERT( r2.ref_count() <= 1, "Not all children of r2 executed" );
    ASSERT( r1.ref_count() > 1, "All children of r1 prematurely executed" );
    r1.wait_for_all();
    ASSERT( r1.ref_count() <= 1, "Not all children of r1 executed" );
    r1.destroy(r1);
    r2.destroy(r2);
}


using tbb::internal::spin_wait_until_eq;

//! Deterministic emulation of a long running task
class LongRunningTask : public tbb::task {
    volatile bool& m_CanProceed;

    tbb::task* execute() __TBB_override {
        spin_wait_until_eq( m_CanProceed, true );
        return NULL;
    }
public:
    LongRunningTask ( volatile bool& canProceed ) : m_CanProceed(canProceed) {}
};

void TestWaitDiscriminativenessWithStealing() {
    if( tbb::tbb_thread::hardware_concurrency() < 2 )
        return;
    REMARK( "testing that task::wait_for_all is specific to the root it is called on (one worker)\n" );
    volatile bool canProceed = false;
    tbb::task_scheduler_init init(2);
    tbb::task &r1 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    tbb::task &r2 = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r1.set_ref_count( 2 );
    r2.set_ref_count( 2 );
    tbb::task& t1 = *new( r1.allocate_child() ) tbb::empty_task;
    tbb::task& t2 = *new( r2.allocate_child() ) LongRunningTask(canProceed);
    tbb::task::spawn(t2);
    tbb::task::spawn(t1);
    r1.wait_for_all();
    ASSERT( r1.ref_count() <= 1, "Not all children of r1 executed" );
    ASSERT( r2.ref_count() == 2, "All children of r2 prematurely executed" );
    canProceed = true;
    r2.wait_for_all();
    ASSERT( r2.ref_count() <= 1, "Not all children of r2 executed" );
    r1.destroy(r1);
    r2.destroy(r2);
}

struct MasterBody : NoAssign, Harness::NoAfterlife {
    static Harness::SpinBarrier my_barrier;

    class BarrenButLongTask : public tbb::task {
        volatile bool& m_Started;
        volatile bool& m_CanProceed;

        tbb::task* execute() __TBB_override {
            m_Started = true;
            spin_wait_until_eq( m_CanProceed, true );
            volatile int k = 0;
            for ( int i = 0; i < 1000000; ++i ) ++k;
            return NULL;
        }
    public:
        BarrenButLongTask ( volatile bool& started, volatile bool& can_proceed )
            : m_Started(started), m_CanProceed(can_proceed)
        {}
    };

    class BinaryRecursiveTask : public tbb::task {
        int m_Depth;

        tbb::task* execute() __TBB_override {
            if( !m_Depth )
                return NULL;
            set_ref_count(3);
            spawn( *new( allocate_child() ) BinaryRecursiveTask(m_Depth - 1) );
            spawn( *new( allocate_child() ) BinaryRecursiveTask(m_Depth - 1) );
            wait_for_all();
            return NULL;
        }

        void note_affinity( affinity_id ) __TBB_override {
            ASSERT( false, "These tasks cannot be stolen" );
        }
    public:
        BinaryRecursiveTask ( int depth_ ) : m_Depth(depth_) {}
    };

    void operator() ( int id ) const {
        if ( id ) {
            tbb::task_scheduler_init init(2);
            volatile bool child_started = false,
                          can_proceed = false;
            tbb::task& r = *new( tbb::task::allocate_root() ) tbb::empty_task;
            r.set_ref_count(2);
            r.spawn( *new(r.allocate_child()) BarrenButLongTask(child_started, can_proceed) );
            spin_wait_until_eq( child_started, true );
            my_barrier.wait();
            can_proceed = true;
            r.wait_for_all();
            r.destroy(r);
        }
        else {
            my_barrier.wait();
            tbb::task_scheduler_init init(1);
            Count = 0;
            int depth = 16;
            BinaryRecursiveTask& r = *new( tbb::task::allocate_root() ) BinaryRecursiveTask(depth);
            tbb::task::spawn_root_and_wait(r);
        }
    }
public:
    MasterBody ( int num_masters ) { my_barrier.initialize(num_masters); }
};

Harness::SpinBarrier MasterBody::my_barrier;

/** Ensures that tasks spawned by a master thread or one of the workers servicing
    it cannot be stolen by another master thread. **/
void TestMastersIsolation ( int p ) {
    // The test requires at least 3-way parallelism to work correctly
    if ( p > 2 && tbb::task_scheduler_init::default_num_threads() >= p ) {
        tbb::task_scheduler_init init(p);
        NativeParallelFor( p, MasterBody(p) );
    }
}

struct waitable_task : tbb::task {
    tbb::task* execute() __TBB_override {
        recycle_as_safe_continuation(); // do not destroy the task after execution
        set_parent(this);               // decrement its own ref_count after completion
        __TBB_Yield();
        return NULL;
    }
};
void TestWaitableTask() {
    waitable_task &wt = *new( tbb::task::allocate_root() ) waitable_task;
    for( int i = 0; i < 100000; i++ ) {
        wt.set_ref_count(2);            // prepare for waiting on it
        wt.spawn(wt);
        if( i&1 ) __TBB_Yield();
        wt.wait_for_all();
    }
    wt.set_parent(NULL);                // prevents assertions and atomics in task::destroy
    tbb::task::destroy(wt);
}

int TestMain () {
#if TBB_USE_EXCEPTIONS
    TestUnconstructibleTask<1>();
    TestUnconstructibleTask<10000>();
#endif
    TestAlignment();
    TestNoteAffinityContext();
    TestDispatchLoopResponsiveness();
    TestWaitDiscriminativenessWithoutStealing();
    TestWaitDiscriminativenessWithStealing();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        TestSpawnChildren( p );
        TestSpawnRootList( p );
        TestSafeContinuation( p );
        TestLeftRecursion( p );
        TestDag( p );
        TestAffinity( p );
        TestUserThread( p );
        TestStealLimit( p );
        TestRelaxedOwnership( p );
        TestMastersIsolation( p );
    }
    TestWaitableTask();
    return Harness::Done;
}
