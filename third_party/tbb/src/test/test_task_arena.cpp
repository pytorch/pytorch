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

#define TBB_PREVIEW_LOCAL_OBSERVER 1
#define __TBB_EXTRA_DEBUG 1

#include <stdexcept>
#include <cstdlib>
#include <cstdio>

#include "harness_fp.h"

#if __TBB_TASK_ISOLATION
// Whitebox stuff for TestIsolatedExecuteNS::ContinuationTest().
// TODO: Consider better approach instead of the whitebox approach.
#define private public
#include "tbb/task.h"
#undef private
#endif /* __TBB_TASK_ISOLATION */

#include "tbb/task_arena.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/enumerable_thread_specific.h"

#include "harness_assert.h"
#include "harness.h"
#include "harness_barrier.h"

#if _MSC_VER
// plays around __TBB_NO_IMPLICIT_LINKAGE. __TBB_LIB_NAME should be defined (in makefiles)
#pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
#endif

//--------------------------------------------------//
// Test that task_arena::initialize and task_arena::terminate work when doing nothing else.
/* maxthread is treated as the biggest possible concurrency level. */
void InitializeAndTerminate( int maxthread ) {
    __TBB_TRY {
        for( int i=0; i<200; ++i ) {
            switch( i&3 ) {
                // Arena is created inactive, initialization is always explicit. Lazy initialization is covered by other test functions.
                // Explicit initialization can either keep the original values or change those.
                // Arena termination can be explicit or implicit (in the destructor).
                // TODO: extend with concurrency level checks if such a method is added.
                default: {
                    tbb::task_arena arena( std::rand() % maxthread + 1 );
                    ASSERT(!arena.is_active(), "arena should not be active until initialized");
                    arena.initialize();
                    ASSERT(arena.is_active(), NULL);
                    arena.terminate();
                    ASSERT(!arena.is_active(), "arena should not be active; it was terminated");
                    break;
                }
                case 0: {
                    tbb::task_arena arena( 1 );
                    ASSERT(!arena.is_active(), "arena should not be active until initialized");
                    arena.initialize( std::rand() % maxthread + 1 ); // change the parameters
                    ASSERT(arena.is_active(), NULL);
                    break;
                }
                case 1: {
                    tbb::task_arena arena( tbb::task_arena::automatic );
                    ASSERT(!arena.is_active(), NULL);
                    arena.initialize();
                    ASSERT(arena.is_active(), NULL);
                    break;
                }
                case 2: {
                    tbb::task_arena arena;
                    ASSERT(!arena.is_active(), "arena should not be active until initialized");
                    arena.initialize( std::rand() % maxthread + 1 );
                    ASSERT(arena.is_active(), NULL);
                    arena.terminate();
                    ASSERT(!arena.is_active(), "arena should not be active; it was terminated");
                    break;
                }
            }
        }
    } __TBB_CATCH( std::runtime_error& error ) {
#if TBB_USE_EXCEPTIONS
        REPORT("ERROR: %s\n", error.what() );
#endif /* TBB_USE_EXCEPTIONS */
    }
}

//--------------------------------------------------//
// Definitions used in more than one test
typedef tbb::blocked_range<int> Range;

// slot_id value: -1 is reserved by current_slot(), -2 is set in on_scheduler_exit() below
static tbb::enumerable_thread_specific<int> local_id, old_id, slot_id(-3);

void ResetTLS() {
    local_id.clear();
    old_id.clear();
    slot_id.clear();
}

class ArenaObserver : public tbb::task_scheduler_observer {
    int myId;               // unique observer/arena id within a test
    int myMaxConcurrency;   // concurrency of the associated arena
    int myNumReservedSlots; // reserved slots in the associated arena
    void on_scheduler_entry( bool is_worker ) __TBB_override {
        int current_index = tbb::this_task_arena::current_thread_index();
        REMARK("a %s #%p is entering arena %d from %d on slot %d\n", is_worker?"worker":"master",
                &local_id.local(), myId, local_id.local(), current_index );
        ASSERT(current_index<(myMaxConcurrency>1?myMaxConcurrency:2), NULL);
        if(is_worker) ASSERT(current_index>=myNumReservedSlots, NULL);

        ASSERT(!old_id.local(), "double call to on_scheduler_entry");
        old_id.local() = local_id.local();
        ASSERT(old_id.local() != myId, "double entry to the same arena");
        local_id.local() = myId;
        slot_id.local() = current_index;
    }
    void on_scheduler_exit( bool is_worker ) __TBB_override {
        REMARK("a %s #%p is leaving arena %d to %d\n", is_worker?"worker":"master",
                &local_id.local(), myId, old_id.local());
        ASSERT(local_id.local() == myId, "nesting of arenas is broken");
        ASSERT(slot_id.local() == tbb::this_task_arena::current_thread_index(), NULL);
         //!deprecated, remove when tbb::task_arena::current_thread_index is removed.
        ASSERT(slot_id.local() == tbb::task_arena::current_thread_index(), NULL);
        slot_id.local() = -2;
        local_id.local() = old_id.local();
        old_id.local() = 0;
    }
public:
    ArenaObserver(tbb::task_arena &a, int maxConcurrency, int numReservedSlots, int id)
        : tbb::task_scheduler_observer(a)
        , myId(id)
        , myMaxConcurrency(maxConcurrency)
        , myNumReservedSlots(numReservedSlots) {
        ASSERT(myId, NULL);
        observe(true);
    }
    ~ArenaObserver () {
        ASSERT(!old_id.local(), "inconsistent observer state");
    }
};

struct IndexTrackingBody { // Must be used together with ArenaObserver
    void operator() ( const Range& ) const {
        ASSERT(slot_id.local() == tbb::this_task_arena::current_thread_index(), NULL);
        //!deprecated, remove when tbb::task_arena::current_thread_index is removed.
        ASSERT(slot_id.local() == tbb::task_arena::current_thread_index(), NULL);
        for ( volatile int i = 0; i < 50000; ++i )
            ;
    }
};

struct AsynchronousWork : NoAssign {
    Harness::SpinBarrier &my_barrier;
    bool my_is_blocking;
    AsynchronousWork(Harness::SpinBarrier &a_barrier, bool blocking = true)
    : my_barrier(a_barrier), my_is_blocking(blocking) {}
    void operator()() const {
        ASSERT(local_id.local() != 0, "not in explicit arena");
        tbb::parallel_for(Range(0,500), IndexTrackingBody(), tbb::simple_partitioner(), *tbb::task::self().group());
        if(my_is_blocking) my_barrier.timed_wait(10); // must be asynchronous to master thread
        else my_barrier.signal_nowait();
    }
};

//--------------------------------------------------//
// Test that task_arenas might be created and used from multiple application threads.
// Also tests arena observers. The parameter p is the index of an app thread running this test.
void TestConcurrentArenasFunc(int idx) {
    tbb::task_arena a1;
    a1.initialize(1,0);
    ArenaObserver o1(a1, 1, 0, idx*2+1); // the last argument is a "unique" observer/arena id for the test
    tbb::task_arena a2(2,1);
    ArenaObserver o2(a2, 2, 1, idx*2+2);
    Harness::SpinBarrier barrier(2);
    AsynchronousWork work(barrier);
    a1.enqueue(work); // put async work
    barrier.timed_wait(10);
    a2.enqueue(work); // another work
    a2.execute(work); // my_barrier.timed_wait(10) inside
    a1.debug_wait_until_empty();
    a2.debug_wait_until_empty();
}

void TestConcurrentArenas(int p) {
    ResetTLS();
    NativeParallelFor( p, &TestConcurrentArenasFunc );
}

//--------------------------------------------------//
// Test multiple application threads working with a single arena at the same time.
class MultipleMastersPart1 : NoAssign {
    tbb::task_arena &my_a;
    Harness::SpinBarrier &my_b1, &my_b2;
public:
    MultipleMastersPart1( tbb::task_arena &a, Harness::SpinBarrier &b1, Harness::SpinBarrier &b2)
        : my_a(a), my_b1(b1), my_b2(b2) {}
    void operator()(int) const {
        my_a.execute(AsynchronousWork(my_b2, /*blocking=*/false));
        my_b1.timed_wait(10);
        // A regression test for bugs 1954 & 1971
        my_a.enqueue(AsynchronousWork(my_b2, /*blocking=*/false));
    }
};

class MultipleMastersPart2 : NoAssign {
    tbb::task_arena &my_a;
    Harness::SpinBarrier &my_b;
public:
    MultipleMastersPart2( tbb::task_arena &a, Harness::SpinBarrier &b) : my_a(a), my_b(b) {}
    void operator()(int) const {
        my_a.execute(AsynchronousWork(my_b, /*blocking=*/false));
    }
};

class MultipleMastersPart3 : NoAssign {
    tbb::task_arena &my_a;
    Harness::SpinBarrier &my_b;

    struct Runner : NoAssign {
        tbb::task* const a_task;
        Runner(tbb::task* const t) : a_task(t) {}
        void operator()() const {
            for ( volatile int i = 0; i < 10000; ++i )
                ;
            a_task->decrement_ref_count();
        }
    };

    struct Waiter : NoAssign {
        tbb::task* const a_task;
        Waiter(tbb::task* const t) : a_task(t) {}
        void operator()() const {
            a_task->wait_for_all();
        }
    };

public:
    MultipleMastersPart3(tbb::task_arena &a, Harness::SpinBarrier &b)
        : my_a(a), my_b(b) {}
    void operator()(int idx) const {
        tbb::empty_task* root_task = new(tbb::task::allocate_root()) tbb::empty_task;
        my_b.timed_wait(10); // increases chances for task_arena initialization contention
        for( int i=0; i<100; ++i) {
            root_task->set_ref_count(2);
            my_a.enqueue(Runner(root_task));
            my_a.execute(Waiter(root_task));
        }
        tbb::task::destroy(*root_task);
        REMARK("Master #%d: job completed, wait for others\n", idx);
        my_b.timed_wait(10);
    }
};

class MultipleMastersPart4 : NoAssign {
    tbb::task_arena &my_a;
    Harness::SpinBarrier &my_b;
    tbb::task_group_context *my_ag;

    struct Getter : NoAssign {
        tbb::task_group_context *& my_g;
        Getter(tbb::task_group_context *&a_g) : my_g(a_g) {}
        void operator()() const {
            my_g = tbb::task::self().group();
        }
    };
    struct Checker : NoAssign {
        tbb::task_group_context *my_g;
        Checker(tbb::task_group_context *a_g) : my_g(a_g) {}
        void operator()() const {
            ASSERT(my_g == tbb::task::self().group(), NULL);
            tbb::task *t = new( tbb::task::allocate_root() ) tbb::empty_task;
            ASSERT(my_g == t->group(), NULL);
            tbb::task::destroy(*t);
        }
    };
    struct NestedChecker : NoAssign {
        const MultipleMastersPart4 &my_body;
        NestedChecker(const MultipleMastersPart4 &b) : my_body(b) {}
        void operator()() const {
            tbb::task_group_context *nested_g = tbb::task::self().group();
            ASSERT(my_body.my_ag != nested_g, NULL);
            tbb::task *t = new( tbb::task::allocate_root() ) tbb::empty_task;
            ASSERT(nested_g == t->group(), NULL);
            tbb::task::destroy(*t);
            my_body.my_a.enqueue(Checker(my_body.my_ag));
        }
    };
public:
    MultipleMastersPart4( tbb::task_arena &a, Harness::SpinBarrier &b) : my_a(a), my_b(b) {
        my_a.execute(Getter(my_ag));
    }
    // NativeParallelFor's functor
    void operator()(int) const {
        my_a.execute(*this);
    }
    // Arena's functor
    void operator()() const {
        Checker check(my_ag);
        check();
        tbb::task_arena nested(1,1);
        nested.execute(NestedChecker(*this)); // change arena
        tbb::parallel_for(Range(0,1),*this); // change group context only
        my_b.timed_wait(10);
        my_a.execute(check);
        check();
    }
    // parallel_for's functor
    void operator()(const Range &) const {
        NestedChecker(*this)();
        my_a.execute(Checker(my_ag)); // restore arena context
    }
};

void TestMultipleMasters(int p) {
    {
        REMARK("multiple masters, part 1\n");
        ResetTLS();
        tbb::task_arena a(1,0);
        a.initialize();
        ArenaObserver o(a, 1, 0, 1);
        Harness::SpinBarrier barrier1(p), barrier2(2*p+1); // each of p threads will submit two tasks signaling the barrier
        NativeParallelFor( p, MultipleMastersPart1(a, barrier1, barrier2) );
        barrier2.timed_wait(10);
        a.debug_wait_until_empty();
    } {
        REMARK("multiple masters, part 2\n");
        ResetTLS();
        tbb::task_arena a(2,1);
        ArenaObserver o(a, 2, 1, 2);
        Harness::SpinBarrier barrier(p+2);
        a.enqueue(AsynchronousWork(barrier, /*blocking=*/true)); // occupy the worker, a regression test for bug 1981
        NativeParallelFor( p, MultipleMastersPart2(a, barrier) );
        barrier.timed_wait(10);
        a.debug_wait_until_empty();
    } {
        // Regression test for the bug 1981 part 2 (task_arena::execute() with wait_for_all for an enqueued task)
        REMARK("multiple masters, part 3: wait_for_all() in execute()\n");
        tbb::task_arena a(p,1);
        Harness::SpinBarrier barrier(p+1); // for masters to avoid endless waiting at least in some runs
        // "Oversubscribe" the arena by 1 master thread
        NativeParallelFor( p+1, MultipleMastersPart3(a, barrier) );
        a.debug_wait_until_empty();
    } {
        int c = p%3? (p%2? p : 2) : 3;
        REMARK("multiple masters, part 4: contexts, arena(%d)\n", c);
        ResetTLS();
        tbb::task_arena a(c, 1);
        ArenaObserver o(a, c, 1, c);
        Harness::SpinBarrier barrier(c);
        MultipleMastersPart4 test(a, barrier);
        NativeParallelFor(p, test);
        a.debug_wait_until_empty();
    }
}

//--------------------------------------------------//
// TODO: explain what TestArenaEntryConsistency does
#include <sstream>
#if TBB_USE_EXCEPTIONS
#include <stdexcept>
#include "tbb/tbb_exception.h"
#endif

struct TestArenaEntryBody : FPModeContext {
    tbb::atomic<int> &my_stage; // each execute increases it
    std::stringstream my_id;
    bool is_caught, is_expected;
    enum { arenaFPMode = 1 };

    TestArenaEntryBody(tbb::atomic<int> &s, int idx, int i)  // init thread-specific instance
    :   FPModeContext(idx+i)
    ,   my_stage(s)
    ,   is_caught(false)
    ,   is_expected( (idx&(1<<i)) != 0 && (TBB_USE_EXCEPTIONS) != 0 )
    {
        my_id << idx << ':' << i << '@';
    }
    void operator()() { // inside task_arena::execute()
        // synchronize with other stages
        int stage = my_stage++;
        int slot = tbb::this_task_arena::current_thread_index();
        ASSERT(slot >= 0 && slot <= 1, "master or the only worker");
        // wait until the third stage is delegated and then starts on slot 0
        while(my_stage < 2+slot) __TBB_Yield();
        // deduct its entry type and put it into id, it helps to find source of a problem
        my_id << (stage < 3 ? (tbb::this_task_arena::current_thread_index()?
                              "delegated_to_worker" : stage < 2? "direct" : "delegated_to_master")
                            : stage == 3? "nested_same_ctx" : "nested_alien_ctx");
        REMARK("running %s\n", my_id.str().c_str());
        AssertFPMode(arenaFPMode);
        if(is_expected)
            __TBB_THROW(std::logic_error(my_id.str()));
        // no code can be put here since exceptions can be thrown
    }
    void on_exception(const char *e) { // outside arena, in catch block
        is_caught = true;
        REMARK("caught %s\n", e);
        ASSERT(my_id.str() == e, NULL);
        assertFPMode();
    }
    void after_execute() { // outside arena and catch block
        REMARK("completing %s\n", my_id.str().c_str() );
        ASSERT(is_caught == is_expected, NULL);
        assertFPMode();
    }
};

class ForEachArenaEntryBody : NoAssign {
    tbb::task_arena &my_a; // expected task_arena(2,1)
    tbb::atomic<int> &my_stage; // each execute increases it
    int my_idx;

public:
    ForEachArenaEntryBody(tbb::task_arena &a, tbb::atomic<int> &c)
    : my_a(a), my_stage(c), my_idx(0) {}

    void test(int idx) {
        my_idx = idx;
        my_stage = 0;
        NativeParallelFor(3, *this); // test cross-arena calls
        ASSERT(my_stage == 3, NULL);
        my_a.execute(*this); // test nested calls
        ASSERT(my_stage == 5, NULL);
    }

    // task_arena functor for nested tests
    void operator()() const {
        test_arena_entry(3); // in current task group context
        tbb::parallel_for(4, 5, *this); // in different context
    }

    // NativeParallelFor & parallel_for functor
    void operator()(int i) const {
        test_arena_entry(i);
    }

private:
    void test_arena_entry(int i) const {
        TestArenaEntryBody scoped_functor(my_stage, my_idx, i);
        __TBB_TRY {
            my_a.execute(scoped_functor);
        }
#if TBB_USE_EXCEPTIONS
        catch(tbb::captured_exception &e) {
            scoped_functor.on_exception(e.what());
            ASSERT_WARNING(TBB_USE_CAPTURED_EXCEPTION, "Caught captured_exception while expecting exact one");
        } catch(std::logic_error &e) {
            scoped_functor.on_exception(e.what());
            ASSERT(!TBB_USE_CAPTURED_EXCEPTION, "Caught exception of wrong type");
        } catch(...) { ASSERT(false, "Unexpected exception type"); }
#endif //TBB_USE_EXCEPTIONS
        scoped_functor.after_execute();
    }
};

void TestArenaEntryConsistency() {
    REMARK("test arena entry consistency\n");

    tbb::task_arena a(2, 1);
    tbb::atomic<int> c;
    ForEachArenaEntryBody body(a, c);

    FPModeContext fp_scope(TestArenaEntryBody::arenaFPMode);
    a.initialize(); // capture FP settings to arena
    fp_scope.setNextFPMode();

    for (int i = 0; i < 100; i++) // not less than 32 = 2^5 of entry types
        body.test(i);
}

//--------------------------------------------------
// Test that the requested degree of concurrency for task_arena is achieved in various conditions
class TestArenaConcurrencyBody : NoAssign {
    tbb::task_arena &my_a;
    int my_max_concurrency;
    int my_reserved_slots;
    Harness::SpinBarrier *my_barrier;
    Harness::SpinBarrier *my_worker_barrier;
public:
    TestArenaConcurrencyBody( tbb::task_arena &a, int max_concurrency, int reserved_slots, Harness::SpinBarrier *b = NULL, Harness::SpinBarrier *wb = NULL )
    : my_a(a), my_max_concurrency(max_concurrency), my_reserved_slots(reserved_slots), my_barrier(b), my_worker_barrier(wb) {}
    // NativeParallelFor's functor
    void operator()( int ) const {
        ASSERT( local_id.local() == 0, "TLS was not cleaned?" );
        local_id.local() = 1;
        my_a.execute( *this );
    }
    // Arena's functor
    void operator()() const {
        ASSERT( tbb::task_arena::current_thread_index() == tbb::this_task_arena::current_thread_index(), NULL );
        int idx = tbb::this_task_arena::current_thread_index();
        ASSERT( idx < (my_max_concurrency > 1 ? my_max_concurrency : 2), NULL );
        ASSERT( my_a.max_concurrency() == tbb::this_task_arena::max_concurrency(), NULL );
        int max_arena_concurrency = tbb::this_task_arena::max_concurrency();
        ASSERT( max_arena_concurrency == my_max_concurrency, NULL );
        if ( my_worker_barrier ) {
            if ( local_id.local() == 1 ) {
                // Master thread in a reserved slot
                ASSERT( idx < my_reserved_slots, "Masters are supposed to use only reserved slots in this test" );
            } else {
                // Worker thread
                ASSERT( idx >= my_reserved_slots, NULL );
                my_worker_barrier->timed_wait( 10 );
            }
        } else if ( my_barrier )
            ASSERT( local_id.local() == 1, "Workers are not supposed to enter the arena in this test" );
        if ( my_barrier ) my_barrier->timed_wait( 10 );
        else Harness::Sleep( 10 );
    }
};

void TestArenaConcurrency( int p, int reserved = 0, int step = 1) {
    for (; reserved <= p; reserved += step) {
        REMARK("TestArenaConcurrency: %d slots, %d reserved\n", p, reserved);
        tbb::task_arena a( p, reserved );
        { // Check concurrency with worker & reserved master threads.
            ResetTLS();
            Harness::SpinBarrier b( p );
            Harness::SpinBarrier wb( p-reserved );
            TestArenaConcurrencyBody test( a, p, reserved, &b, &wb );
            for ( int i = reserved; i < p; ++i )
                a.enqueue( test );
            if ( reserved==1 )
                test( 0 ); // calls execute()
            else
                NativeParallelFor( reserved, test );
            a.debug_wait_until_empty();
        } { // Check if multiple masters alone can achieve maximum concurrency.
            ResetTLS();
            Harness::SpinBarrier b( p );
            NativeParallelFor( p, TestArenaConcurrencyBody( a, p, reserved, &b ) );
            a.debug_wait_until_empty();
        } { // Check oversubscription by masters.
            ResetTLS();
            NativeParallelFor( 2*p, TestArenaConcurrencyBody( a, p, reserved ) );
            a.debug_wait_until_empty();
        }
    }
}

//--------------------------------------------------//
// Test creation/initialization of a task_arena that references an existing arena (aka attach).
// This part of the test uses the knowledge of task_arena internals

typedef tbb::interface7::internal::task_arena_base task_arena_internals;

struct TaskArenaValidator : public task_arena_internals {
    int my_slot_at_construction;
    TaskArenaValidator( const task_arena_internals& other )
    : task_arena_internals(other) /*copies the internal state of other*/ {
        my_slot_at_construction = tbb::this_task_arena::current_thread_index();
    }
    // Inspect the internal state
    int concurrency() { return my_max_concurrency; }
    int reserved_for_masters() { return (int)my_master_slots; }

    // This method should be called in task_arena::execute() for a captured arena
    // by the same thread that created the validator.
    void operator()() {
        ASSERT( tbb::this_task_arena::current_thread_index()==my_slot_at_construction,
                "Current thread index has changed since the validator construction" );
        //!deprecated
        ASSERT( tbb::task_arena::current_thread_index()==my_slot_at_construction,
                "Current thread index has changed since the validator construction" );
    }
};

void ValidateAttachedArena( tbb::task_arena& arena, bool expect_activated,
                            int expect_concurrency, int expect_masters ) {
    ASSERT( arena.is_active()==expect_activated, "Unexpected activation state" );
    if( arena.is_active() ) {
        TaskArenaValidator validator( arena );
        ASSERT( validator.concurrency()==expect_concurrency, "Unexpected arena size" );
        ASSERT( validator.reserved_for_masters()==expect_masters, "Unexpected # of reserved slots" );
        if ( tbb::this_task_arena::current_thread_index() != tbb::task_arena::not_initialized ) {
            ASSERT( tbb::task_arena::current_thread_index() >= 0 &&
                tbb::this_task_arena::current_thread_index() >= 0, NULL);
            // for threads already in arena, check that the thread index remains the same
            arena.execute( validator );
        } else { // not_initialized
            // Test the deprecated method
            ASSERT( tbb::task_arena::current_thread_index()==-1, NULL);
        }

        // Ideally, there should be a check for having the same internal arena object,
        // but that object is not easily accessible for implicit arenas.
    }
}

struct TestAttachBody : NoAssign {
    mutable int my_idx; // safe to modify and use within the NativeParallelFor functor
    const int maxthread;
    TestAttachBody( int max_thr ) : maxthread(max_thr) {}

    // The functor body for NativeParallelFor
    void operator()( int idx ) const {
        my_idx = idx;
        int default_threads = tbb::task_scheduler_init::default_num_threads();

        tbb::task_arena arena = tbb::task_arena( tbb::task_arena::attach() );
        ValidateAttachedArena( arena, false, -1, -1 ); // Nothing yet to attach to

        { // attach to an arena created via task_scheduler_init
            tbb::task_scheduler_init init( idx+1 );

            tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
            ValidateAttachedArena( arena2, true, idx+1, 1 );

            arena.initialize( tbb::task_arena::attach() );
        }
        ValidateAttachedArena( arena, true, idx+1, 1 );

        arena.terminate();
        ValidateAttachedArena( arena, false, -1, -1 );

        // Check default behavior when attach cannot succeed
        switch (idx%2) {
        case 0:
            { // construct as attached, then initialize
                tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
                ValidateAttachedArena( arena2, false, -1, -1 );
                arena2.initialize(); // must be initialized with default parameters
                ValidateAttachedArena( arena2, true, default_threads, 1 );
            }
            break;
        case 1:
            { // default-construct, then initialize as attached
                tbb::task_arena arena2;
                ValidateAttachedArena( arena2, false, -1, -1 );
                arena2.initialize( tbb::task_arena::attach() ); // must use default parameters
                ValidateAttachedArena( arena2, true, default_threads, 1 );
            }
            break;
        } // switch

        // attach to an auto-initialized arena
        tbb::empty_task& tsk = *new (tbb::task::allocate_root()) tbb::empty_task;
        tbb::task::spawn_root_and_wait(tsk);
        tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
        ValidateAttachedArena( arena2, true, default_threads, 1 );

        // attach to another task_arena
        arena.initialize( maxthread, min(maxthread,idx) );
        arena.execute( *this );
    }

    // The functor body for task_arena::execute above
    void operator()() const {
        tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
        ValidateAttachedArena( arena2, true, maxthread, min(maxthread,my_idx) );
    }

    // The functor body for tbb::parallel_for
    void operator()( const Range& r ) const {
        for( int i = r.begin(); i<r.end(); ++i ) {
            tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
            ValidateAttachedArena( arena2, true, maxthread+1, 1 ); // +1 to match initialization in TestMain
        }
    }
};

void TestAttach( int maxthread ) {
    REMARK( "Testing attached task_arenas\n" );
    // Externally concurrent, but no concurrency within a thread
    NativeParallelFor( max(maxthread,4), TestAttachBody( maxthread ) );
    // Concurrent within the current arena; may also serve as a stress test
    tbb::parallel_for( Range(0,10000*maxthread), TestAttachBody( maxthread ) );
}

//--------------------------------------------------//
// Test that task_arena::enqueue does not tolerate a non-const functor.
// TODO: can it be reworked as SFINAE-based compile-time check?
struct TestFunctor {
    void operator()() { ASSERT( false, "Non-const operator called" ); }
    void operator()() const { /* library requires this overload only */ }
};

void TestConstantFunctorRequirement() {
    tbb::task_arena a;
    TestFunctor tf;
    a.enqueue( tf );
#if __TBB_TASK_PRIORITY
    a.enqueue( tf, tbb::priority_normal );
#endif
}
//--------------------------------------------------//
#if __TBB_TASK_ISOLATION
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_invoke.h"
// Test this_task_arena::isolate
namespace TestIsolatedExecuteNS {
    //--------------------------------------------------//
    template <typename NestedPartitioner>
    class NestedParFor : NoAssign {
    public:
        NestedParFor() {}
        void operator()() const {
            NestedPartitioner p;
            tbb::parallel_for( 0, 10, Harness::DummyBody( 10 ), p );
        }
    };

    template <typename NestedPartitioner>
    class ParForBody : NoAssign {
        bool myOuterIsolation;
        tbb::enumerable_thread_specific<int> &myEts;
        tbb::atomic<bool> &myIsStolen;
    public:
        ParForBody( bool outer_isolation, tbb::enumerable_thread_specific<int> &ets, tbb::atomic<bool> &is_stolen )
            : myOuterIsolation( outer_isolation ), myEts( ets ), myIsStolen( is_stolen ) {}
        void operator()( int ) const {
            int &e = myEts.local();
            if ( e++ > 0 ) myIsStolen = true;
            if ( myOuterIsolation )
                NestedParFor<NestedPartitioner>()();
            else
                tbb::this_task_arena::isolate( NestedParFor<NestedPartitioner>() );
            --e;
        }
    };

    template <typename OuterPartitioner, typename NestedPartitioner>
    class OuterParFor : NoAssign {
        bool myOuterIsolation;
        tbb::atomic<bool> &myIsStolen;
    public:
        OuterParFor( bool outer_isolation, tbb::atomic<bool> &is_stolen ) : myOuterIsolation( outer_isolation ), myIsStolen( is_stolen ) {}
        void operator()() const {
            tbb::enumerable_thread_specific<int> ets( 0 );
            OuterPartitioner p;
            tbb::parallel_for( 0, 1000, ParForBody<NestedPartitioner>( myOuterIsolation, ets, myIsStolen ), p );
        }
    };

    template <typename OuterPartitioner, typename NestedPartitioner>
    void TwoLoopsTest( bool outer_isolation ) {
        tbb::atomic<bool> is_stolen;
        is_stolen = false;
        const int max_repeats = 100;
        if ( outer_isolation ) {
            for ( int i = 0; i <= max_repeats; ++i ) {
                tbb::this_task_arena::isolate( OuterParFor<OuterPartitioner, NestedPartitioner>( outer_isolation, is_stolen ) );
                if ( is_stolen ) break;
            }
            ASSERT_WARNING( is_stolen, "isolate() should not block stealing on nested levels without isolation" );
        } else {
            for ( int i = 0; i <= max_repeats; ++i ) {
                OuterParFor<OuterPartitioner, NestedPartitioner>( outer_isolation, is_stolen )();
            }
            ASSERT( !is_stolen, "isolate() on nested levels should prevent stealing from outer leves" );
        }
    }

    void TwoLoopsTest( bool outer_isolation ) {
        TwoLoopsTest<tbb::simple_partitioner, tbb::simple_partitioner>( outer_isolation );
        TwoLoopsTest<tbb::simple_partitioner, tbb::affinity_partitioner>( outer_isolation );
        TwoLoopsTest<tbb::affinity_partitioner, tbb::simple_partitioner>( outer_isolation );
        TwoLoopsTest<tbb::affinity_partitioner, tbb::affinity_partitioner>( outer_isolation );
    }

    void TwoLoopsTest() {
        TwoLoopsTest( true );
        TwoLoopsTest( false );
    }
    //--------------------------------------------------//
    class HeavyMixTestBody : NoAssign {
        tbb::enumerable_thread_specific<Harness::FastRandom>& myRandom;
        tbb::enumerable_thread_specific<int>& myIsolatedLevel;
        int myNestedLevel;
        bool myHighPriority;

        template <typename Partitioner, typename Body>
        static void RunTwoBodies( Harness::FastRandom& rnd, const Body &body, Partitioner& p, tbb::task_group_context* ctx = NULL ) {
            if ( rnd.get() % 2 )
                if  (ctx )
                    tbb::parallel_for( 0, 2, body, p, *ctx );
                else
                    tbb::parallel_for( 0, 2, body, p );
            else
                tbb::parallel_invoke( body, body );
        }

        template <typename Partitioner>
        class IsolatedBody : NoAssign {
            const HeavyMixTestBody &myHeavyMixTestBody;
            Partitioner &myPartitioner;
        public:
            IsolatedBody( const HeavyMixTestBody &body, Partitioner &partitioner )
                : myHeavyMixTestBody( body ), myPartitioner( partitioner ) {}
            void operator()() const {
                RunTwoBodies( myHeavyMixTestBody.myRandom.local(),
                    HeavyMixTestBody( myHeavyMixTestBody.myRandom, myHeavyMixTestBody.myIsolatedLevel,
                        myHeavyMixTestBody.myNestedLevel + 1, myHeavyMixTestBody.myHighPriority ),
                    myPartitioner );
            }
        };

        template <typename Partitioner>
        void RunNextLevel( Harness::FastRandom& rnd, int &isolated_level ) const {
            Partitioner p;
            switch ( rnd.get() % 3 ) {
                case 0: {
                    // No features
                    tbb::task_group_context ctx;
                    if ( myHighPriority )
                        ctx.set_priority( tbb::priority_high );
                    RunTwoBodies( rnd, HeavyMixTestBody(myRandom, myIsolatedLevel, myNestedLevel + 1, myHighPriority), p, &ctx );
                    break;
                }
                case 1: {
                    // High priority
                    tbb::task_group_context ctx;
                    ctx.set_priority( tbb::priority_high );
                    RunTwoBodies( rnd, HeavyMixTestBody(myRandom, myIsolatedLevel, myNestedLevel + 1, true), p, &ctx );
                    break;
                }
                case 2: {
                    // Isolation
                    int previous_isolation = isolated_level;
                    isolated_level = myNestedLevel;
                    tbb::this_task_arena::isolate( IsolatedBody<Partitioner>( *this, p ) );
                    isolated_level = previous_isolation;
                    break;
                }
            }
        }
    public:
        HeavyMixTestBody( tbb::enumerable_thread_specific<Harness::FastRandom>& random,
            tbb::enumerable_thread_specific<int>& isolated_level, int nested_level, bool high_priority )
            : myRandom( random ), myIsolatedLevel( isolated_level )
            , myNestedLevel( nested_level ), myHighPriority( high_priority ) {}
        void operator()() const {
            int &isolated_level = myIsolatedLevel.local();
            ASSERT( myNestedLevel > isolated_level, "The outer-level task should not be stolen on isolated level" );
            if ( myNestedLevel == 20 )
                return;
            Harness::FastRandom &rnd = myRandom.local();
            if ( rnd.get() % 2 == 1 ) {
                RunNextLevel<tbb::auto_partitioner>( rnd, isolated_level );
            } else {
                RunNextLevel<tbb::affinity_partitioner>( rnd, isolated_level );
            }
        }
        void operator()(int) const {
            this->operator()();
        }
    };

    struct RandomInitializer {
        Harness::FastRandom operator()() {
            return Harness::FastRandom( tbb::this_task_arena::current_thread_index() );
        }
    };

    void HeavyMixTest() {
        tbb::task_scheduler_init init( tbb::task_scheduler_init::default_num_threads() < 3 ? 3 : tbb::task_scheduler_init::automatic );
        RandomInitializer init_random;
        tbb::enumerable_thread_specific<Harness::FastRandom> random( init_random );
        tbb::enumerable_thread_specific<int> isolated_level( 0 );
        for ( int i = 0; i < 5; ++i ) {
            HeavyMixTestBody b( random, isolated_level, 1, false );
            b( 0 );
            REMARK( "\rHeavyMixTest: %d of 10", i+1 );
        }
        REMARK( "\n" );
    }
    //--------------------------------------------------//
    struct ContinuationTestReduceBody : NoAssign {
        tbb::internal::isolation_tag myIsolation;
        ContinuationTestReduceBody( tbb::internal::isolation_tag isolation ) : myIsolation( isolation ) {}
        ContinuationTestReduceBody( ContinuationTestReduceBody& b, tbb::split ) : myIsolation( b.myIsolation ) {}
        void operator()( tbb::blocked_range<int> ) {}
        void join( ContinuationTestReduceBody& ) {
            tbb::internal::isolation_tag isolation = tbb::task::self().prefix().isolation;
            ASSERT( isolation == myIsolation, "The continuations should preserve children's isolation" );
        }
    };
    struct ContinuationTestIsolated {
        void operator()() const {
            ContinuationTestReduceBody b( tbb::task::self().prefix().isolation );
            tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( 0, 100 ), b );
        }
    };
    struct ContinuationTestParForBody : NoAssign {
        tbb::enumerable_thread_specific<int> &myEts;
    public:
        ContinuationTestParForBody( tbb::enumerable_thread_specific<int> &ets ) : myEts( ets ){}
        void operator()( int ) const {
            int &e = myEts.local();
            ++e;
            ASSERT( e==1, "The task is stolen on isolated level" );
            tbb::this_task_arena::isolate( ContinuationTestIsolated() );
            --e;
        }
    };
    void ContinuationTest() {
        for ( int i = 0; i < 5; ++i ) {
            tbb::enumerable_thread_specific<int> myEts;
            tbb::parallel_for( 0, 100, ContinuationTestParForBody( myEts ), tbb::simple_partitioner() );
        }
    }
    //--------------------------------------------------//
#if TBB_USE_EXCEPTIONS
    struct MyException {};
    struct IsolatedBodyThrowsException {
        void operator()() const {
            __TBB_THROW( MyException() );
        }
    };
    struct ExceptionTestBody : NoAssign {
        tbb::enumerable_thread_specific<int>& myEts;
        tbb::atomic<bool>& myIsStolen;
        ExceptionTestBody( tbb::enumerable_thread_specific<int>& ets, tbb::atomic<bool>& is_stolen )
            : myEts( ets ), myIsStolen( is_stolen ) {}
        void operator()( int i ) const {
            try {
                tbb::this_task_arena::isolate( IsolatedBodyThrowsException() );
                ASSERT( false, "The exception has been lost" );
            }
            catch ( MyException ) {}
            catch ( ... ) {
                ASSERT( false, "Unexpected exception" );
            }
            // Check that nested algorithms can steal outer-level tasks
            int &e = myEts.local();
            if ( e++ > 0 ) myIsStolen = true;
            // work imbalance increases chances for stealing
            tbb::parallel_for( 0, 10+i, Harness::DummyBody( 100 ) );
            --e;
        }
    };

#endif /* TBB_USE_EXCEPTIONS */
    void ExceptionTest() {
#if TBB_USE_EXCEPTIONS
        tbb::enumerable_thread_specific<int> ets;
        tbb::atomic<bool> is_stolen;
        is_stolen = false;
        for ( int i = 0; i<10; ++i ) {
            tbb::parallel_for( 0, 1000, ExceptionTestBody( ets, is_stolen ) );
            if ( is_stolen ) break;
        }
        ASSERT( is_stolen, "isolate should not affect non-isolated work" );
#endif /* TBB_USE_EXCEPTIONS */
    }

    struct NonConstBody {
        unsigned int state;
        void operator()() {
            state ^= ~0u;
        }
    };

    void TestNonConstBody() {
        NonConstBody body;
        body.state = 0x6c97d5ed;
        tbb::this_task_arena::isolate(body);
        ASSERT(body.state == 0x93682a12, "The wrong state");
    }
}

void TestIsolatedExecute() {
    // At least 3 threads (owner + 2 thieves) are required to reproduce a situation when the owner steals outer
    // level task on a nested level. If we have only one thief then it will execute outer level tasks first and
    // the owner will not have a possibility to steal outer level tasks.
    int num_threads = min( tbb::task_scheduler_init::default_num_threads(), 3 );
    {
        // Too many threads require too many work to reproduce the stealing from outer level.
        tbb::task_scheduler_init init( max(num_threads, 7) );
        TestIsolatedExecuteNS::TwoLoopsTest();
        TestIsolatedExecuteNS::HeavyMixTest();
        TestIsolatedExecuteNS::ContinuationTest();
        TestIsolatedExecuteNS::ExceptionTest();
    }
    tbb::task_scheduler_init init(num_threads);
    TestIsolatedExecuteNS::HeavyMixTest();
    TestIsolatedExecuteNS::ContinuationTest();
    TestIsolatedExecuteNS::TestNonConstBody();
}
#endif /* __TBB_TASK_ISOLATION */
//--------------------------------------------------//
//--------------------------------------------------//

class TestDelegatedSpawnWaitBody : NoAssign {
    tbb::task_arena &my_a;
    Harness::SpinBarrier &my_b1, &my_b2;

    struct Spawner : NoAssign {
        tbb::task* const a_task;
        Spawner(tbb::task* const t) : a_task(t) {}
        void operator()() const {
            tbb::task::spawn( *new(a_task->allocate_child()) tbb::empty_task );
        }
    };

    struct Waiter : NoAssign {
        tbb::task* const a_task;
        Waiter(tbb::task* const t) : a_task(t) {}
        void operator()() const {
            a_task->wait_for_all();
        }
    };

public:
    TestDelegatedSpawnWaitBody( tbb::task_arena &a, Harness::SpinBarrier &b1, Harness::SpinBarrier &b2)
        : my_a(a), my_b1(b1), my_b2(b2) {}
    // NativeParallelFor's functor
    void operator()(int idx) const {
        if ( idx==0 ) { // thread 0 works in the arena, thread 1 waits for it (to prevent test hang)
            for( int i=0; i<2; ++i ) my_a.enqueue(*this); // tasks to sync with workers
            tbb::empty_task* root_task = new(tbb::task::allocate_root()) tbb::empty_task;
            root_task->set_ref_count(100001);
            my_b1.timed_wait(10); // sync with the workers
            for( int i=0; i<100000; ++i) {
                my_a.execute(Spawner(root_task));
            }
            my_a.execute(Waiter(root_task));
            tbb::task::destroy(*root_task);
        }
        my_b2.timed_wait(10); // sync both threads
    }
    // Arena's functor
    void operator()() const {
        my_b1.timed_wait(10); // sync with the arena master
    }
};

void TestDelegatedSpawnWait() {
    // Regression test for a bug with missed wakeup notification from a delegated task
    REMARK( "Testing delegated spawn & wait\n" );
    tbb::task_arena a(2,0);
    a.initialize();
    Harness::SpinBarrier barrier1(3), barrier2(2);
    NativeParallelFor( 2, TestDelegatedSpawnWaitBody(a, barrier1, barrier2) );
    a.debug_wait_until_empty();
}

class TestMultipleWaitsArenaWait : NoAssign {
public:
    TestMultipleWaitsArenaWait( int idx, int bunch_size, int num_tasks, tbb::task** waiters, tbb::atomic<int>& processed )
        : my_idx( idx ), my_bunch_size( bunch_size ), my_num_tasks(num_tasks), my_waiters( waiters ), my_processed( processed ) {}
    void operator()() const {
        ++my_processed;
        // Wait for all tasks
        if ( my_idx < my_num_tasks )
            my_waiters[my_idx]->wait_for_all();
        // Signal waiting tasks
        if ( my_idx >= my_bunch_size )
            my_waiters[my_idx-my_bunch_size]->decrement_ref_count();
    }
private:
    int my_idx;
    int my_bunch_size;
    int my_num_tasks;
    tbb::task** my_waiters;
    tbb::atomic<int>& my_processed;
};

class TestMultipleWaitsThreadBody : NoAssign {
public:
    TestMultipleWaitsThreadBody( int bunch_size, int num_tasks, tbb::task_arena& a, tbb::task** waiters, tbb::atomic<int>& processed )
        : my_bunch_size( bunch_size ), my_num_tasks( num_tasks ), my_arena( a ), my_waiters( waiters ), my_processed( processed ) {}
    void operator()( int idx ) const {
        my_arena.execute( TestMultipleWaitsArenaWait( idx, my_bunch_size, my_num_tasks, my_waiters, my_processed ) );
        --my_processed;
    }
private:
    int my_bunch_size;
    int my_num_tasks;
    tbb::task_arena& my_arena;
    tbb::task** my_waiters;
    tbb::atomic<int>& my_processed;
};

#include "tbb/tbb_thread.h"

void TestMultipleWaits( int num_threads, int num_bunches, int bunch_size ) {
    tbb::task_arena a( num_threads );
    const int num_tasks = (num_bunches-1)*bunch_size;
    tbb::task** tasks = new tbb::task*[num_tasks];
    for ( int i = 0; i<num_tasks; ++i )
        tasks[i] = new (tbb::task::allocate_root()) tbb::empty_task();
    tbb::atomic<int> processed;
    processed = 0;
    for ( int repeats = 0; repeats<10; ++repeats ) {
        int idx = 0;
        for ( int bunch = 0; bunch < num_bunches-1; ++bunch ) {
            // Sync with the previous bunch of tasks to prevent "false" nested dependicies (when a nested task waits for an outer task).
            while ( processed < bunch*bunch_size ) __TBB_Yield();
            // Run the bunch of threads/tasks that depend on the next bunch of threads/tasks.
            for ( int i = 0; i<bunch_size; ++i ) {
                tasks[idx]->set_ref_count( 2 );
                tbb::tbb_thread( TestMultipleWaitsThreadBody( bunch_size, num_tasks, a, tasks, processed ), idx++ ).detach();
            }
        }
        // No sync because the threads of the last bunch do not call wait_for_all.
        // Run the last bunch of threads.
        for ( int i = 0; i<bunch_size; ++i )
            tbb::tbb_thread( TestMultipleWaitsThreadBody( bunch_size, num_tasks, a, tasks, processed ), idx++ ).detach();
        while ( processed ) __TBB_Yield();
    }
    for ( int i = 0; i<num_tasks; ++i )
        tbb::task::destroy( *tasks[i] );
    delete[] tasks;
}

void TestMultipleWaits() {
    REMARK( "Testing multiple waits\n" );
    // Limit the number of threads to prevent heavy oversubscription.
    const int max_threads = min( 16, tbb::task_scheduler_init::default_num_threads() );

    Harness::FastRandom rnd(1234);
    for ( int threads = 1; threads <= max_threads; threads += max( threads/2, 1 ) ) {
        for ( int i = 0; i<3; ++i ) {
            const int num_bunches = 3 + rnd.get()%3;
            const int bunch_size = max_threads + rnd.get()%max_threads;
            TestMultipleWaits( threads, num_bunches, bunch_size );
        }
    }
}
//--------------------------------------------------//
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"

void TestSmallStackSize() {
    tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic,
        tbb::global_control::active_value(tbb::global_control::thread_stack_size) / 2 );
    // The test produces the warning (not a error) if fails. So the test is run many times
    // to make the log annoying (to force to consider it as an error).
    for (int i = 0; i < 100; ++i) {
        tbb::task_arena a;
        a.initialize();
    }
}
//--------------------------------------------------//
#if __TBB_CPP11_RVALUE_REF_PRESENT
namespace TestMoveSemanticsNS {
    struct TestFunctor {
        void operator()() const {};
    };

    struct MoveOnlyFunctor : MoveOnly, TestFunctor {
        MoveOnlyFunctor() : MoveOnly() {};
        MoveOnlyFunctor(MoveOnlyFunctor&& other) : MoveOnly(std::move(other)) {};
    };

    struct MovePreferableFunctor : Movable, TestFunctor {
        MovePreferableFunctor() : Movable() {};
        MovePreferableFunctor(MovePreferableFunctor&& other) : Movable( std::move(other) ) {};
        MovePreferableFunctor(const MovePreferableFunctor& other) : Movable(other) {};
    };

    struct NoMoveNoCopyFunctor : NoCopy, TestFunctor {
        NoMoveNoCopyFunctor() : NoCopy() {};
        // mv ctor is not allowed as cp ctor from parent NoCopy
    private:
        NoMoveNoCopyFunctor(NoMoveNoCopyFunctor&&);
    };


    void TestFunctors() {
        tbb::task_arena ta;
        MovePreferableFunctor mpf;
        // execute() doesn't have any copies or moves of arguments inside the impl
        ta.execute( NoMoveNoCopyFunctor() );

        ta.enqueue( MoveOnlyFunctor() );
        ta.enqueue( mpf );
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();
        ta.enqueue( std::move(mpf) );
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
#if __TBB_TASK_PRIORITY
        ta.enqueue( MoveOnlyFunctor(), tbb::priority_normal );
        ta.enqueue( mpf, tbb::priority_normal );
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();
        ta.enqueue( std::move(mpf), tbb::priority_normal );
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
#endif
    }
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

void TestMoveSemantics() {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveSemanticsNS::TestFunctors();
#else
    REPORT("Known issue: move support tests are skipped.\n");
#endif
}
//--------------------------------------------------//
#if __TBB_CPP11_DECLTYPE_PRESENT && !__TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN
#include <vector>
#include "harness_state_trackable.h"

namespace TestReturnValueNS {
    struct noDefaultTag {};
    class ReturnType : public Harness::StateTrackable<> {
        static const int SIZE = 42;
        std::vector<int> data;
    public:
        ReturnType(noDefaultTag) : Harness::StateTrackable<>(0) {}
#if !__TBB_IMPLICIT_MOVE_PRESENT
        // Define copy constructor to test that it is never called
        ReturnType(const ReturnType& r) : Harness::StateTrackable<>(r), data(r.data) {}
        ReturnType(ReturnType&& r) : Harness::StateTrackable<>(std::move(r)), data(std::move(r.data)) {}
#endif
        void fill() {
            for (int i = 0; i < SIZE; ++i)
                data.push_back(i);
        }
        void check() {
            ASSERT(data.size() == unsigned(SIZE), NULL);
            for (int i = 0; i < SIZE; ++i)
                ASSERT(data[i] == i, NULL);
            Harness::StateTrackableCounters::counters_t& cnts = Harness::StateTrackableCounters::counters;
            ASSERT((cnts[Harness::StateTrackableBase::DefaultInitialized] == 0), NULL);
            ASSERT(cnts[Harness::StateTrackableBase::DirectInitialized] == 1, NULL);
            std::size_t copied = cnts[Harness::StateTrackableBase::CopyInitialized];
            std::size_t moved = cnts[Harness::StateTrackableBase::MoveInitialized];
            ASSERT(cnts[Harness::StateTrackableBase::Destroyed] == copied + moved, NULL);
            // The number of copies/moves should not exceed 3: function return, store to an internal storage,
            // acquire internal storage.
#if __TBB_CPP11_RVALUE_REF_PRESENT
            ASSERT(copied == 0 && moved <=3, NULL);
#else
            ASSERT(copied <= 3 && moved == 0, NULL);
#endif
        }
    };

    template <typename R>
    R function() {
        noDefaultTag tag;
        R r(tag);
        r.fill();
        return r;
    }

    template <>
    void function<void>() {}

    template <typename R>
    struct Functor {
        R operator()() const {
            return function<R>();
        }
    };

    tbb::task_arena& arena() {
        static tbb::task_arena a;
        return a;
    }

    template <typename F>
    void TestExecute(F &f) {
        Harness::StateTrackableCounters::reset();
        ReturnType r = arena().execute(f);
        r.check();
    }

    template <typename F>
    void TestExecute(const F &f) {
        Harness::StateTrackableCounters::reset();
        ReturnType r = arena().execute(f);
        r.check();
    }
#if TBB_PREVIEW_TASK_ISOLATION
    template <typename F>
    void TestIsolate(F &f) {
        Harness::StateTrackableCounters::reset();
        ReturnType r = tbb::this_task_arena::isolate(f);
        r.check();
    }

    template <typename F>
    void TestIsolate(const F &f) {
        Harness::StateTrackableCounters::reset();
        ReturnType r = tbb::this_task_arena::isolate(f);
        r.check();
    }
#endif

    void Test() {
        TestExecute(Functor<ReturnType>());
        Functor<ReturnType> f1;
        TestExecute(f1);
        TestExecute(function<ReturnType>);

        arena().execute(Functor<void>());
        Functor<void> f2;
        arena().execute(f2);
        arena().execute(function<void>);
#if TBB_PREVIEW_TASK_ISOLATION
        TestIsolate(Functor<ReturnType>());
        TestIsolate(f1);
        TestIsolate(function<ReturnType>);
        tbb::this_task_arena::isolate(Functor<void>());
        tbb::this_task_arena::isolate(f2);
        tbb::this_task_arena::isolate(function<void>);
#endif
    }
}
#endif /* __TBB_CPP11_DECLTYPE_PRESENT */

void TestReturnValue() {
#if __TBB_CPP11_DECLTYPE_PRESENT && !__TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN
    TestReturnValueNS::Test();
#endif
}
//--------------------------------------------------//
void TestConcurrentFunctionality(int min_thread_num = MinThread, int max_thread_num = MaxThread) {
    InitializeAndTerminate(max_thread_num);
    for (int p = min_thread_num; p <= max_thread_num; ++p) {
        REMARK("testing with %d threads\n", p);
        TestConcurrentArenas(p);
        TestMultipleMasters(p);
        TestArenaConcurrency(p);
    }
}
//--------------------------------------------------//
struct DefaultCreatedWorkersAmountBody {
    int my_threadnum;
    DefaultCreatedWorkersAmountBody(int threadnum) : my_threadnum(threadnum) {}
    void operator()(int) const {
        ASSERT(my_threadnum == tbb::this_task_arena::max_concurrency(), "concurrency level is not equal specified threadnum");
        ASSERT(tbb::this_task_arena::current_thread_index() < tbb::this_task_arena::max_concurrency(), "amount of created threads is more than specified by default");
        local_id.local() = 1;
        Harness::Sleep(1);
    }
};

struct NativeParallelForBody {
    int my_thread_num;
    int iterations;
    NativeParallelForBody(int thread_num, int multiplier = 100) : my_thread_num(thread_num), iterations(multiplier * thread_num) {}
    void operator()(int idx) const {
        ASSERT(idx == 0, "more than 1 thread is going to reset TLS");
        ResetTLS();
        tbb::parallel_for(0, iterations, DefaultCreatedWorkersAmountBody(my_thread_num), tbb::simple_partitioner());
        ASSERT(local_id.size() == size_t(my_thread_num), "amount of created threads is not equal to default num");
    }
};

void TestDefaultCreatedWorkersAmount() {
    NativeParallelFor(1, NativeParallelForBody(tbb::task_scheduler_init::default_num_threads()));
}

void TestAbilityToCreateWorkers(int thread_num) {
    tbb::task_scheduler_init init_market_with_necessary_amount_plus_one(thread_num);
    // Checks only some part of reserved-master threads amount:
    // 0 and 1 reserved threads are important cases but it is also needed
    // to collect some statistic data with other amount and to not consume
    // whole test sesion time checking each amount
    TestArenaConcurrency(thread_num - 1, 0, int(thread_num / 2.72));
    TestArenaConcurrency(thread_num, 1, int(thread_num / 3.14));
}

void TestDefaultWorkersLimit() {
    TestDefaultCreatedWorkersAmount();
    // Shared RML might limit the number of workers even if you specify the limits
    // by the reason of (default_concurrency==max_concurrency) for shared RML
#ifndef RML_USE_WCRM
    TestAbilityToCreateWorkers(256);
#endif
}
//--------------------------------------------------//

int TestMain () {
#if __TBB_TASK_ISOLATION
    TestIsolatedExecute();
#endif /* __TBB_TASK_ISOLATION */
    TestSmallStackSize();
    TestDefaultWorkersLimit();
    // The test uses up to MaxThread workers (in arenas with no master thread),
    // so the runtime should be initialized appropriately.
    tbb::task_scheduler_init init_market_p_plus_one(MaxThread + 1);
    TestConcurrentFunctionality();
    TestArenaEntryConsistency();
    TestAttach(MaxThread);
    TestConstantFunctorRequirement();
    TestDelegatedSpawnWait();
    TestMultipleWaits();
    TestMoveSemantics();
    TestReturnValue();
    return Harness::Done;
}

