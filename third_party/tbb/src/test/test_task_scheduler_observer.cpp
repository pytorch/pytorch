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

#if __TBB_CPF_BUILD
#define TEST_SLEEP_PERMISSION 1
#define TBB_USE_PREVIEW_BINARY 1
#endif
// undefine __TBB_CPF_BUILD to simulate user's setup
#undef __TBB_CPF_BUILD

#define TBB_PREVIEW_LOCAL_OBSERVER 1

#include "tbb/tbb_config.h"
#include "harness.h"

#if __TBB_SCHEDULER_OBSERVER
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"
#include "tbb/task.h"
#include "tbb/enumerable_thread_specific.h"
#include "../tbb/tls.h"
#include "tbb/tick_count.h"
#include "harness_barrier.h"

#if _MSC_VER && __TBB_NO_IMPLICIT_LINKAGE
// plays around __TBB_NO_IMPLICIT_LINKAGE. __TBB_LIB_NAME should be defined (in makefiles)
    #pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
#endif

const int MaxFlagIndex = sizeof(uintptr_t)*8-1;

struct ObserverStats {
    tbb::atomic<int> m_entries;
    tbb::atomic<int> m_exits;
    tbb::atomic<int> m_workerEntries;
    tbb::atomic<int> m_workerSleeps;
    tbb::atomic<int> m_workerExits;

    void Reset () {
        m_entries = m_exits = m_workerEntries = m_workerSleeps = m_workerExits = 0;
    }

    void operator += ( const ObserverStats& s ) {
        m_entries += s.m_entries;
        m_exits += s.m_exits;
        m_workerEntries += s.m_workerEntries;
        m_workerSleeps += s.m_workerSleeps;
        m_workerExits += s.m_workerExits;
    }
};

struct ThreadState {
    uintptr_t m_flags;
    tbb::task_scheduler_observer *m_dyingObserver;
    uintptr_t m_maySleepCalls;
    bool m_canSleep;
    bool m_isMaster;
    ThreadState() { reset(); }
    void reset() {
        m_maySleepCalls = m_flags = 0;
        m_dyingObserver = NULL;
        m_canSleep = m_isMaster = false;
    }
    static ThreadState &get();
};

tbb::enumerable_thread_specific<ThreadState> theLocalState;
tbb::internal::tls<intptr_t> theThreadPrivate;

ThreadState &ThreadState::get() {
    bool exists;
    ThreadState& state = theLocalState.local(exists);
    // ETS will not detect that a thread was allocated with the same id as a destroyed thread
    if( exists && theThreadPrivate.get() == 0 ) state.reset();
    theThreadPrivate = 1; // mark thread constructed
    return state;
}

static ObserverStats theStats;
static tbb::atomic<int> theNumObservers;

const int P = min( tbb::task_scheduler_init::default_num_threads(), (int)sizeof(int) * CHAR_BIT );

enum TestMode {
    //! Ensure timely workers destruction in order to guarantee all exit notification are fired.
    tmSynchronized = 1,
    //! Use local observer.
    tmLocalObservation = 2,
    //! Observer causes autoinitialization of the scheduler
    tmAutoinitialization = 4,
    //! test may_sleep
    tmLeavingControl = 8
};

uintptr_t theTestMode,
          thePrevMode = 0;

class MyObserver : public tbb::task_scheduler_observer, public ObserverStats {
    uintptr_t m_flag;
    tbb::atomic<int> m_leave_ticket;
    tbb::atomic<bool> m_dying;

    void on_scheduler_entry( bool is_worker ) __TBB_override {
        ThreadState& state = ThreadState::get();
        ASSERT( is_worker==!state.m_isMaster, NULL );
        if ( theTestMode & tmLeavingControl )
            ASSERT( m_leave_ticket, NULL );
        if ( thePrevMode & tmSynchronized ) {
            ASSERT( !(state.m_flags & m_flag), "Observer repeatedly invoked for the same thread" );
            if ( theTestMode & tmLocalObservation )
                ASSERT( !state.m_flags, "Observer locality breached" );
        }
        if ( m_dying && theTestMode & tmLocalObservation ) {
            // In case of local observation a worker may enter the arena after
            // the wait for lagging on_entry calls in the MyObserver destructor
            // succeeds but before its base class tbb::task_scheduler_observer
            // destructor removes it from the internal list maintained by the
            // task scheduler. This will result in on_entry notification without,
            // subsequent on_exit as the observer is likely to be destroyed before
            // the worker discovers that the arena is empty and leaves it.
            //
            // To prevent statistics distortion, ignore the notifications for
            // observers about to be destroyed.
            ASSERT( !state.m_dyingObserver || state.m_dyingObserver != this || thePrevMode & tmSynchronized, NULL );
            state.m_dyingObserver = this;
            return;
        }
        state.m_dyingObserver = NULL;
        ++m_entries;
        state.m_flags |= m_flag;
        if ( is_worker )
            ++m_workerEntries;
    }
    void on_scheduler_exit( bool is_worker ) __TBB_override {
        ThreadState& state = ThreadState::get();
        ASSERT( is_worker==!state.m_isMaster, NULL );
        if ( m_dying && state.m_dyingObserver ) {
            ASSERT( state.m_dyingObserver == this, "Exit without entry (for a dying observer)" );
            state.m_dyingObserver = NULL;
            return;
        }
        ASSERT( state.m_flags & m_flag, "Exit without entry" );
        state.m_flags &= ~m_flag;
        ++m_exits;
        if ( is_worker )
            ++m_workerExits;
    }
    bool may_sleep() __TBB_override {
        ThreadState& state = ThreadState::get();
        ++state.m_maySleepCalls;
        Harness::Sleep(10);     // helps to reproduce the issues
        ASSERT( !state.m_isMaster, NULL );
        if( m_dying ) {         // check the anti-starvation logic
            return keep_awake;  // thread should exit despite the return value
        }
        if( state.m_canSleep ) {// the permission for sleep was previously received
            // though, it is an important check for the test, we still do not guarantee this condition
            ASSERT_WARNING( !(theTestMode & tmLeavingControl), "may_sleep() called again after leaving permission was granted once, check if repeated");
            return allow_sleep;
        }
        // note, may_sleep can be called before on_entry()
        if( !(theTestMode & tmLeavingControl) || m_leave_ticket.fetch_and_store(-1) > 0 ) {
            state.m_canSleep = true;
            ++m_workerSleeps;
            return allow_sleep;
        }
        return keep_awake;
    }
public:
    // the method is called before the work in new arena starts enabling the leaving test mode
    // in this mode may_sleep() does not allow a thread to fall asleep unless permitted below
    void enable_leaving_test() {
        ASSERT(theTestMode & tmLeavingControl, NULL);
        m_leave_ticket.store<tbb::relaxed>(-1);
        ASSERT(!is_observing(), NULL);
        observe(true);
    }

    // the work is just done in the only arena, assume workers start entering may_sleep
    void test_leaving() {
#if TEST_SLEEP_PERMISSION
        if( !(theTestMode & tmLeavingControl) )
            return; // second call to the test TODO: extend the test for the second round as well
        REMARK( "Testing may_sleep()\n");
        ASSERT( !m_workerSleeps, "permission for sleep was given before the test starts?");
        ASSERT( (theTestMode & tmSynchronized) && m_workerEntries >= P-1, "test_leaving assumes full subscription of the only arena");
        for ( int j = 0; j < m_workerEntries; j++ ) {
            REMARK( "Round %d: entries %d, sleeps %d\n", j, (int)m_workerEntries, (int)m_workerSleeps );
            ASSERT( m_leave_ticket == -1, "unexpected mode, signal was not consumed by a worker?" );
            m_leave_ticket = 1; // dismiss one
            double n_seconds = 10;
            (Harness::TimedWaitWhileEq(n_seconds))(m_workerSleeps, j);
            ASSERT( n_seconds >= 0, "Time out while waiting for a worker to call may_sleep for the first time");
            __TBB_Yield();
        }
        // the first time this method is called the work will be executed again,
        // the next time time, the scheduler will start shutting down
        theTestMode &= ~tmLeavingControl;
        m_leave_ticket = m_workerSleeps = 0; // reset for the next round
#endif
    }

    MyObserver( uintptr_t flag )
        : tbb::task_scheduler_observer(theTestMode & tmLocalObservation ? true : false)
        , m_flag(flag)
    {
        m_leave_ticket.store<tbb::relaxed>(0);
        ++theNumObservers;
        Reset();
        m_dying = false;
        // Local observer causes automatic scheduler initialization
        // in the current thread, so here, we must postpone the activation.
        if ( !(theTestMode & tmLocalObservation) && !(theTestMode & tmLeavingControl) )
            observe(true);
    }

    ~MyObserver () {
        m_dying = true;
        ASSERT( m_exits <= m_entries, NULL );
        if ( theTestMode & tmSynchronized ) {
            tbb::tick_count t0 = tbb::tick_count::now();
            while ( m_exits < m_entries && (tbb::tick_count::now() - t0).seconds() < 5 )
                Harness::Sleep(10);
            if ( m_exits < m_entries )
                REPORT( "Warning: Entry/exit count mismatch (%d, %d). Observer is broken or machine is overloaded.\n", (int)m_entries, (int)m_exits );
        }
        theStats += *this;
        --theNumObservers;
        // it is recommended to disable observation before destructor of the base class starts,
        // otherwise it can lead to concurrent notification callback on partly destroyed object,
        // which in turn can harm (in addition) if derived class has new virtual methods.
        // This class has no, and for test purposes we rely on implementation failsafe mechanism.
        //observe(false);
    }
}; // class MyObserver

Harness::SpinBarrier theGlobalBarrier;
bool theGlobalBarrierActive = true;

class FibTask : public tbb::task {
    const int N;
    uintptr_t m_flag;
    MyObserver &m_observer;
public:
    FibTask( int n, uintptr_t flags, MyObserver &obs ) : N(n), m_flag(flags), m_observer(obs) {}

    tbb::task* execute() __TBB_override {
        ThreadState& s = ThreadState::get();
        ASSERT( !(~s.m_flags & m_flag), NULL );
        if( N < 2 )
            return NULL;
        bool globalBarrierActive = false;
        if ( s.m_isMaster ) {
            if ( theGlobalBarrierActive ) {
                // This is the root task. Its N is equal to the number of threads.
                // Spawn a task for each worker.
                set_ref_count(N);
                for ( int i = 1; i < N; ++i )
                    spawn( *new( allocate_child() ) FibTask(20, m_flag, m_observer) );
                if ( theTestMode & tmSynchronized ) {
                    theGlobalBarrier.wait();
                    ASSERT( m_observer.m_entries >= N, "Wrong number of on_entry calls after the first barrier" );
                    // All the spawned tasks have been stolen by workers.
                    // Now wait for workers to spawn some more tasks for this thread to steal back.
                    theGlobalBarrier.wait();
                    ASSERT( !theGlobalBarrierActive, "Workers are expected to have reset this flag" );
                }
                else
                    theGlobalBarrierActive = false;
                wait_for_all();
                return NULL;
            }
        }
        else {
            if ( theGlobalBarrierActive ) {
                if ( theTestMode & tmSynchronized ) {
                    theGlobalBarrier.wait();
                    globalBarrierActive = true;
                }
                theGlobalBarrierActive = false;
            }
        }
        set_ref_count(3);
        spawn( *new( allocate_child() ) FibTask(N-1, m_flag, m_observer) );
        spawn( *new( allocate_child() ) FibTask(N-2, m_flag, m_observer) );
        if ( globalBarrierActive ) {
            // It's the first task executed by a worker. Release the master thread.
            theGlobalBarrier.wait();
        }
        wait_for_all();
        return NULL;
    }
}; // class FibTask

Harness::SpinBarrier theMasterBarrier;

class TestBody {
    int m_numThreads;
public:
    TestBody( int numThreads ) : m_numThreads(numThreads) {}

    void operator()( int i ) const {
        ThreadState &state = ThreadState::get();
        ASSERT( !state.m_isMaster, "should be newly initialized thread");
        state.m_isMaster = true;
        uintptr_t f = i <= MaxFlagIndex ? 1<<i : 0;
        MyObserver o(f);
        if ( theTestMode & tmSynchronized )
            theMasterBarrier.wait();
        // when mode is local observation but not synchronized and when num threads == default
        if ( theTestMode & tmAutoinitialization )
            o.observe(true); // test autoinitialization can be done by observer
        // when mode is synchronized observation and when num threads == default
        if ( theTestMode & tmLeavingControl )
            o.enable_leaving_test();
        // Observer in enabled state must outlive the scheduler to ensure that
        // all exit notifications are called.
        tbb::task_scheduler_init init(m_numThreads);
        // when local & non-autoinitialized observation mode
        if ( theTestMode & tmLocalObservation )
            o.observe(true);
        for ( int j = 0; j < 2; ++j ) {
            tbb::task &t = *new( tbb::task::allocate_root() ) FibTask(m_numThreads, f, o);
            tbb::task::spawn_root_and_wait(t);
            if ( theTestMode & tmLeavingControl )
                o.test_leaving();
            thePrevMode = theTestMode;
        }
    }
}; // class TestBody

void TestObserver( int M, int T, uintptr_t testMode ) {
    theLocalState.clear();
    theStats.Reset();
    theGlobalBarrierActive = true;
    theTestMode = testMode;
    NativeParallelFor( M, TestBody(T) );
    // When T (number of threads in arena, i.e. master + workers) is less than P
    // (hardware concurrency), more than T-1 workers can visit the same arena. This
    // is possible in case of imbalance or when other arenas are activated/deactivated
    // concurrently).
    ASSERT( !theNumObservers, "Unexpected alive observer(s)" );
    REMARK( "Entries %d / %d, exits %d\n", (int)theStats.m_entries, (int)theStats.m_workerEntries, (int)theStats.m_exits );
    if ( testMode & tmSynchronized ) {
        if ( testMode & tmLocalObservation ) {
            ASSERT( theStats.m_entries >= M * T, "Too few on_entry calls" );
            ASSERT( theStats.m_workerEntries >= M * (T - 1), "Too few worker entries" );
        }
        else {
            ASSERT( theStats.m_entries >= M * M * T, "Too few on_entry calls" );
            ASSERT( theStats.m_entries <= M * (P + 1), "Too many on_entry calls" );
            ASSERT( theStats.m_workerEntries >= M * M * (T - 1), "Too few worker entries" );
            ASSERT( theStats.m_workerEntries <= M * (P - 1), "Too many worker entries" );
        }
        ASSERT( theStats.m_entries == theStats.m_exits, "Entries/exits mismatch" );
    }
    else {
        ASSERT( theStats.m_entries >= M, "Too few on_entry calls" );
        ASSERT( theStats.m_exits >= M || (testMode & tmAutoinitialization), "Too few on_exit calls" );
        if ( !(testMode & tmLocalObservation) ) {
            ASSERT( theStats.m_entries <= M * M * P, "Too many on_entry calls" );
            ASSERT( theStats.m_exits <= M * M * T, "Too many on_exit calls" );
        }
        ASSERT( theStats.m_entries >= theStats.m_exits, "More exits than entries" );
    }
}

int TestMain () {
    if ( P < 2 )
        return Harness::Skipped;
    theNumObservers = 0;
    // Fully- and under-utilized mode
    for ( int M = 1; M < P; M <<= 1 ) {
        if ( M > P/2 ) {
            ASSERT( P & (P-1), "Can get here only in case of non power of two cores" );
            M = P/2;
            if ( M==1 || (M & (M-1)) )
                break; // Already tested this configuration
        }
        int T = P / M;
        ASSERT( T > 1, NULL );
        REMARK( "Masters: %d; Arena size: %d\n", M, T );
        theMasterBarrier.initialize(M);
        theGlobalBarrier.initialize(M * T);
        TestObserver(M, T, 0);
        TestObserver(M, T, tmSynchronized | tmLocalObservation );
        TestObserver(M, T, tmSynchronized | ( T==P? tmLeavingControl : 0));
        // keep tmAutoInitialization the last, as it does not release worker threads
        TestObserver(M, T, tmLocalObservation | ( T==P? tmAutoinitialization : 0) );
    }
    // Oversubscribed mode
    for ( int i = 0; i < 4; ++i ) {
        REMARK( "Masters: %d; Arena size: %d\n", P-1, P );
        TestObserver(P-1, P, 0);
        TestObserver(P-1, P, tmLocalObservation);
    }
    Harness::Sleep(20);
    return Harness::Done;
}

#else /* !__TBB_SCHEDULER_OBSERVER */

int TestMain () {
    return Harness::Skipped;
}
#endif /* !__TBB_SCHEDULER_OBSERVER */
