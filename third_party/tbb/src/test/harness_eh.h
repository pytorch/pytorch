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

#include <typeinfo>
#include "tbb/tbb_exception.h"
#include "tbb/atomic.h"
#if USE_TASK_SCHEDULER_OBSERVER
#include "tbb/task_scheduler_observer.h"
#endif
#include "harness.h"
#include "harness_concurrency_tracker.h"

int g_NumThreads = 0;
Harness::tid_t  g_Master = 0;
const char * g_Orig_Wakeup_Msg = "Missed wakeup or machine is overloaded?";
const char * g_Wakeup_Msg = g_Orig_Wakeup_Msg;

tbb::atomic<intptr_t> g_CurExecuted,
                      g_ExecutedAtLastCatch,
                      g_ExecutedAtFirstCatch,
                      g_ExceptionsThrown,
                      g_MasterExecutedThrow,     // number of times master entered exception code
                      g_NonMasterExecutedThrow,  // number of times nonmaster entered exception code
                      g_PipelinesStarted;
volatile bool g_ExceptionCaught = false,
              g_UnknownException = false;

#if USE_TASK_SCHEDULER_OBSERVER
tbb::atomic<intptr_t> g_ActualMaxThreads;
tbb::atomic<intptr_t> g_ActualCurrentThreads;
#endif

volatile bool g_ThrowException = true,
         // g_Flog is true for nested construct tests with catches (exceptions are not allowed to
         // propagate to the tbb construct itself.)
              g_Flog = false,
              g_MasterExecuted = false,
              g_NonMasterExecuted = false;

bool    g_ExceptionInMaster = false;
bool    g_SolitaryException = false;
bool    g_NestedPipelines   = false;

//! Number of exceptions propagated into the user code (i.e. intercepted by the tests)
tbb::atomic<intptr_t> g_NumExceptionsCaught;

//-----------------------------------------------------------

#if USE_TASK_SCHEDULER_OBSERVER
class eh_test_observer : public tbb::task_scheduler_observer {
public:
    void on_scheduler_entry(bool is_worker) __TBB_override {
        if(is_worker) {  // we've already counted the master
            size_t p = ++g_ActualCurrentThreads;
            size_t q = g_ActualMaxThreads;
            while(q < p) {
                q = g_ActualMaxThreads.compare_and_swap(p,q);
            }
        }
        else {
            // size_t q = g_ActualMaxThreads;
        }
    }
    void on_scheduler_exit(bool is_worker) __TBB_override {
        if(is_worker) {
            --g_ActualCurrentThreads;
        }
    }
};
#endif
//-----------------------------------------------------------

inline void ResetEhGlobals ( bool throwException = true, bool flog = false ) {
    Harness::ConcurrencyTracker::Reset();
    g_CurExecuted = g_ExecutedAtLastCatch = g_ExecutedAtFirstCatch = 0;
    g_ExceptionCaught = false;
    g_UnknownException = false;
    g_NestedPipelines = false;
    g_ThrowException = throwException;
    g_MasterExecutedThrow = 0;
    g_NonMasterExecutedThrow = 0;
    g_Flog = flog;
    g_MasterExecuted = false;
    g_NonMasterExecuted = false;
#if USE_TASK_SCHEDULER_OBSERVER
    g_ActualMaxThreads = 1;  // count master
    g_ActualCurrentThreads = 1;  // count master
#endif
    g_ExceptionsThrown = g_NumExceptionsCaught = g_PipelinesStarted = 0;
}

#if TBB_USE_EXCEPTIONS
class test_exception : public std::exception {
    const char* my_description;
public:
    test_exception ( const char* description ) : my_description(description) {}

    const char* what() const throw() __TBB_override { return my_description; }
};

class solitary_test_exception : public test_exception {
public:
    solitary_test_exception ( const char* description ) : test_exception(description) {}
};

#if TBB_USE_CAPTURED_EXCEPTION
    typedef tbb::captured_exception PropagatedException;
    #define EXCEPTION_NAME(e) e.name()
#else
    typedef test_exception PropagatedException;
    #define EXCEPTION_NAME(e) typeid(e).name()
#endif

#define EXCEPTION_DESCR "Test exception"

#if HARNESS_EH_SIMPLE_MODE

static void ThrowTestException () {
    ++g_ExceptionsThrown;
    throw test_exception(EXCEPTION_DESCR);
}

#else /* !HARNESS_EH_SIMPLE_MODE */

static void ThrowTestException ( intptr_t threshold ) {
    bool inMaster = (Harness::CurrentTid() == g_Master);
    if ( !g_ThrowException ||   // if we're not supposed to throw
            (!g_Flog &&         // if we're not catching throw in bodies and
             (g_ExceptionInMaster ^ inMaster)) ) { // we're the master and not expected to throw
              // or are the master and the master is not the one to throw (??)
        return;
    }
    while ( Existed() < threshold )
        __TBB_Yield();
    if ( !g_SolitaryException ) {
        ++g_ExceptionsThrown;
        if(inMaster) ++g_MasterExecutedThrow; else ++g_NonMasterExecutedThrow;
        throw test_exception(EXCEPTION_DESCR);
    }
    // g_SolitaryException == true
    if(g_NestedPipelines) {
        // only throw exception if we have started at least two inner pipelines
        // else return
        if(g_PipelinesStarted >= 3) {
            if ( g_ExceptionsThrown.compare_and_swap(1, 0) == 0 )  {
                if(inMaster) ++g_MasterExecutedThrow; else ++g_NonMasterExecutedThrow;
                throw solitary_test_exception(EXCEPTION_DESCR);
            }
        }
    }
    else {
        if ( g_ExceptionsThrown.compare_and_swap(1, 0) == 0 )  {
            if(inMaster) ++g_MasterExecutedThrow; else ++g_NonMasterExecutedThrow;
            throw solitary_test_exception(EXCEPTION_DESCR);
        }
    }
}
#endif /* !HARNESS_EH_SIMPLE_MODE */

#define UPDATE_COUNTS()     \
    { \
        ++g_CurExecuted; \
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true; \
        else g_NonMasterExecuted = true; \
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled; \
    }

#define CATCH()     \
    } catch ( PropagatedException& e ) { \
        g_ExecutedAtFirstCatch.compare_and_swap(g_CurExecuted,0); \
        g_ExecutedAtLastCatch = g_CurExecuted; \
        ASSERT( e.what(), "Empty what() string" );  \
        ASSERT (__TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(EXCEPTION_NAME(e), (g_SolitaryException ? typeid(solitary_test_exception) : typeid(test_exception)).name() ) == 0, "Unexpected original exception name"); \
        ASSERT (__TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR) == 0, "Unexpected original exception info"); \
        g_ExceptionCaught = l_ExceptionCaughtAtCurrentLevel = true; \
        ++g_NumExceptionsCaught; \
    } catch ( tbb::tbb_exception& e ) { \
        REPORT("Unexpected %s\n", e.name()); \
        ASSERT (g_UnknownException && !g_UnknownException, "Unexpected tbb::tbb_exception" ); \
    } catch ( std::exception& e ) { \
        REPORT("Unexpected %s\n", typeid(e).name()); \
        ASSERT (g_UnknownException && !g_UnknownException, "Unexpected std::exception" ); \
    } catch ( ... ) { \
        g_ExceptionCaught = l_ExceptionCaughtAtCurrentLevel = true; \
        g_UnknownException = unknownException = true; \
    } \
    if ( !g_SolitaryException ) \
        REMARK_ONCE ("Multiple exceptions mode: %d throws", (intptr_t)g_ExceptionsThrown);

#define ASSERT_EXCEPTION() \
    { \
        ASSERT (!g_ExceptionsThrown || g_ExceptionCaught, "throw without catch"); \
        ASSERT (!g_ExceptionCaught || g_ExceptionsThrown, "catch without throw"); \
        ASSERT (g_ExceptionCaught || (g_ExceptionInMaster && !g_MasterExecutedThrow) || (!g_ExceptionInMaster && !g_NonMasterExecutedThrow), "no exception occurred"); \
        ASSERT (__TBB_EXCEPTION_TYPE_INFO_BROKEN || !g_UnknownException, "unknown exception was caught"); \
    }

#define CATCH_AND_ASSERT() \
    CATCH() \
    ASSERT_EXCEPTION()

#else /* !TBB_USE_EXCEPTIONS */

inline void ThrowTestException ( intptr_t ) {}

#endif /* !TBB_USE_EXCEPTIONS */

#define TRY()   \
    bool l_ExceptionCaughtAtCurrentLevel = false, unknownException = false;    \
    __TBB_TRY {

// "l_ExceptionCaughtAtCurrentLevel || unknownException" is used only to "touch" otherwise unused local variables
#define CATCH_AND_FAIL() } __TBB_CATCH(...) { \
        ASSERT (false, "Cancelling tasks must not cause any exceptions");    \
        (void)(l_ExceptionCaughtAtCurrentLevel && unknownException);                        \
    }

const int c_Timeout = 1000000;

void WaitUntilConcurrencyPeaks ( int expected_peak ) {
    if ( g_Flog )
        return;
    int n = 0;
retry:
    while ( ++n < c_Timeout && (int)Harness::ConcurrencyTracker::PeakParallelism() < expected_peak )
        __TBB_Yield();
#if USE_TASK_SCHEDULER_OBSERVER
    ASSERT_WARNING( g_NumThreads == g_ActualMaxThreads, "Library did not provide sufficient threads");
#endif
    ASSERT_WARNING(n < c_Timeout,g_Wakeup_Msg);
    // Workaround in case a missed wakeup takes place
    if ( n == c_Timeout ) {
        tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task();
        r.spawn(r);
        n = 0;
        goto retry;
    }
}

inline void WaitUntilConcurrencyPeaks () { WaitUntilConcurrencyPeaks(g_NumThreads); }

inline bool IsMaster() {
    return Harness::CurrentTid() == g_Master;
}

inline bool IsThrowingThread() {
    return g_ExceptionInMaster ^ IsMaster() ? true : false;
}

class CancellatorTask : public tbb::task {
    static volatile bool s_Ready;
    tbb::task_group_context &m_groupToCancel;
    intptr_t m_cancellationThreshold;

    tbb::task* execute () __TBB_override {
        Harness::ConcurrencyTracker ct;
        s_Ready = true;
        while ( g_CurExecuted < m_cancellationThreshold )
            __TBB_Yield();
        m_groupToCancel.cancel_group_execution();
        g_ExecutedAtLastCatch = g_CurExecuted;
        return NULL;
    }
public:
    CancellatorTask ( tbb::task_group_context& ctx, intptr_t threshold )
        : m_groupToCancel(ctx), m_cancellationThreshold(threshold)
    {
        s_Ready = false;
    }

    static void Reset () { s_Ready = false; }

    static bool WaitUntilReady () {
        const intptr_t limit = 10000000;
        intptr_t n = 0;
        do {
            __TBB_Yield();
        } while( !s_Ready && ++n < limit );
        // should yield once, then continue if Cancellator is ready.
        ASSERT( s_Ready || n == limit, NULL );
        return s_Ready;
    }
};

volatile bool CancellatorTask::s_Ready = false;

template<class LauncherTaskT, class CancellatorTaskT>
void RunCancellationTest ( intptr_t threshold = 1 )
{
    tbb::task_group_context  ctx;
    tbb::empty_task &r = *new( tbb::task::allocate_root(ctx) ) tbb::empty_task;
    r.set_ref_count(3);
    r.spawn( *new( r.allocate_child() ) CancellatorTaskT(ctx, threshold) );
    __TBB_Yield();
    r.spawn( *new( r.allocate_child() ) LauncherTaskT(ctx) );
    TRY();
        r.wait_for_all();
    CATCH_AND_FAIL();
    r.destroy(r);
}
