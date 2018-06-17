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

//Concurrency scheduler is not supported by Windows* new UI apps
//TODO: check whether we can test anything here
#include "tbb/tbb_config.h"
#if !__TBB_WIN8UI_SUPPORT
#ifndef TBBTEST_USE_TBB
    #define TBBTEST_USE_TBB 1
#endif
#else
    #define TBBTEST_USE_TBB 0
    #undef __TBB_TASK_GROUP_CONTEXT
    #define __TBB_TASK_GROUP_CONTEXT 0
#endif

#if !TBBTEST_USE_TBB
    #if defined(_MSC_VER) && _MSC_VER < 1600
        #ifdef TBBTEST_USE_TBB
            #undef TBBTEST_USE_TBB
        #endif
        #define TBBTEST_USE_TBB 1
    #endif
#endif

#if TBBTEST_USE_TBB

    #include "tbb/compat/ppl.h"
    #include "tbb/task_scheduler_init.h"

    #if _MSC_VER
        typedef tbb::internal::uint32_t uint_t;
    #else
        typedef uint32_t uint_t;
    #endif

#else /* !TBBTEST_USE_TBB */

    #if defined(_MSC_VER)
    #pragma warning(disable: 4100 4180)
    #endif

    #include <ppl.h>

    typedef unsigned int uint_t;

    // Bug in this ConcRT version results in task_group::wait() rethrowing
    // internal cancellation exception propagated by the scheduler from the nesting
    // task group.
    #define __TBB_SILENT_CANCELLATION_BROKEN  (_MSC_VER == 1600)

#endif /* !TBBTEST_USE_TBB */

#if __TBB_TASK_GROUP_CONTEXT

#include "tbb/atomic.h"
#include "tbb/aligned_space.h"
#include "harness.h"
#include "harness_concurrency_tracker.h"

unsigned g_MaxConcurrency = 0;

typedef tbb::atomic<uint_t> atomic_t;
typedef Concurrency::task_handle<void(*)()> handle_type;

//------------------------------------------------------------------------
// Tests for the thread safety of the task_group manipulations
//------------------------------------------------------------------------

#include "harness_barrier.h"

enum SharingMode {
    VagabondGroup = 1,
    ParallelWait = 2
};

class  SharedGroupBodyImpl : NoCopy, Harness::NoAfterlife {
    static const uint_t c_numTasks0 = 4096,
                        c_numTasks1 = 1024;

    const uint_t m_numThreads;
    const uint_t m_sharingMode;

    Concurrency::task_group *m_taskGroup;
    atomic_t m_tasksSpawned,
             m_threadsReady;
    Harness::SpinBarrier m_barrier;

    static atomic_t s_tasksExecuted;

    struct TaskFunctor {
        SharedGroupBodyImpl *m_pOwner;
        void operator () () const {
            if ( m_pOwner->m_sharingMode & ParallelWait ) {
                while ( Harness::ConcurrencyTracker::PeakParallelism() < m_pOwner->m_numThreads )
                    __TBB_Yield();
            }
            ++s_tasksExecuted;
        }
    };

    TaskFunctor m_taskFunctor;

    void Spawn ( uint_t numTasks ) {
        for ( uint_t i = 0; i < numTasks; ++i ) {
            ++m_tasksSpawned;
            Harness::ConcurrencyTracker ct;
            m_taskGroup->run( m_taskFunctor );
        }
        ++m_threadsReady;
    }

    void DeleteTaskGroup () {
        delete m_taskGroup;
        m_taskGroup = NULL;
    }

    void Wait () {
        while ( m_threadsReady != m_numThreads )
            __TBB_Yield();
        const uint_t numSpawned = c_numTasks0 + c_numTasks1 * (m_numThreads - 1);
        ASSERT ( m_tasksSpawned == numSpawned, "Wrong number of spawned tasks. The test is broken" );
        REMARK("Max spawning parallelism is %u out of %u\n", Harness::ConcurrencyTracker::PeakParallelism(), g_MaxConcurrency);
        if ( m_sharingMode & ParallelWait ) {
            m_barrier.wait( &Harness::ConcurrencyTracker::Reset );
            {
                Harness::ConcurrencyTracker ct;
                m_taskGroup->wait();
            }
            if ( Harness::ConcurrencyTracker::PeakParallelism() == 1 )
                REPORT ( "Warning: No parallel waiting detected in TestParallelWait\n" );
            m_barrier.wait();
        }
        else
            m_taskGroup->wait();
        ASSERT ( m_tasksSpawned == numSpawned, "No tasks should be spawned after wait starts. The test is broken" );
        ASSERT ( s_tasksExecuted == numSpawned, "Not all spawned tasks were executed" );
    }

public:
    SharedGroupBodyImpl ( uint_t numThreads, uint_t sharingMode = 0 )
        : m_numThreads(numThreads)
        , m_sharingMode(sharingMode)
        , m_taskGroup(NULL)
        , m_barrier(numThreads)
    {
        ASSERT ( m_numThreads > 1, "SharedGroupBody tests require concurrency" );
        ASSERT ( !(m_sharingMode & VagabondGroup) || m_numThreads == 2, "In vagabond mode SharedGroupBody must be used with 2 threads only" );
        Harness::ConcurrencyTracker::Reset();
        s_tasksExecuted = 0;
        m_tasksSpawned = 0;
        m_threadsReady = 0;
        m_taskFunctor.m_pOwner = this;
    }

    void Run ( uint_t idx ) {
#if TBBTEST_USE_TBB
        tbb::task_scheduler_init init(g_MaxConcurrency);
#endif
        AssertLive();
        if ( idx == 0 ) {
            ASSERT ( !m_taskGroup && !m_tasksSpawned, "SharedGroupBody must be reset before reuse");
            m_taskGroup = new Concurrency::task_group;
            Spawn( c_numTasks0 );
            Wait();
            if ( m_sharingMode & VagabondGroup )
                m_barrier.wait();
            else
                DeleteTaskGroup();
        }
        else {
            while ( m_tasksSpawned == 0 )
                __TBB_Yield();
            ASSERT ( m_taskGroup, "Task group is not initialized");
            Spawn (c_numTasks1);
            if ( m_sharingMode & ParallelWait )
                Wait();
            if ( m_sharingMode & VagabondGroup ) {
                ASSERT ( idx == 1, "In vagabond mode SharedGroupBody must be used with 2 threads only" );
                m_barrier.wait();
                DeleteTaskGroup();
            }
        }
        AssertLive();
    }
};

atomic_t SharedGroupBodyImpl::s_tasksExecuted;

class  SharedGroupBody : NoAssign, Harness::NoAfterlife {
    bool m_bOwner;
    SharedGroupBodyImpl *m_pImpl;
public:
    SharedGroupBody ( uint_t numThreads, uint_t sharingMode = 0 )
        : m_bOwner(true)
        , m_pImpl( new SharedGroupBodyImpl(numThreads, sharingMode) )
    {}
    SharedGroupBody ( const SharedGroupBody& src )
        : NoAssign()
        , Harness::NoAfterlife()
        , m_bOwner(false)
        , m_pImpl(src.m_pImpl)
    {}
    ~SharedGroupBody () {
        if ( m_bOwner )
            delete m_pImpl;
    }
    void operator() ( uint_t idx ) const { m_pImpl->Run(idx); }
};

class RunAndWaitSyncronizationTestBody {
    Harness::SpinBarrier& m_barrier;
    tbb::atomic<bool>& m_completed;
    tbb::task_group& m_tg;
public:
    RunAndWaitSyncronizationTestBody(Harness::SpinBarrier& barrier, tbb::atomic<bool>& completed, tbb::task_group& tg)
        : m_barrier(barrier), m_completed(completed), m_tg(tg) {}

    void operator()() const {
        m_barrier.wait();
        for (volatile int i = 0; i < 100000; ++i) {}
        m_completed = true;
    }

    void operator()(int id) const {
        if (id == 0) {
            m_tg.run_and_wait(*this);
        } else {
            m_barrier.wait();
            m_tg.wait();
            ASSERT(m_completed, "A concurrent waiter has left the wait method earlier than work has finished");
        }
    }
};

void TestParallelSpawn () {
    NativeParallelFor( g_MaxConcurrency, SharedGroupBody(g_MaxConcurrency) );
}

void TestParallelWait () {
    NativeParallelFor( g_MaxConcurrency, SharedGroupBody(g_MaxConcurrency, ParallelWait) );

    Harness::SpinBarrier barrier(g_MaxConcurrency);
    tbb::atomic<bool> completed;
    completed = false;
    tbb::task_group tg;
    RunAndWaitSyncronizationTestBody b(barrier, completed, tg);
    NativeParallelFor( g_MaxConcurrency, b );
}

// Tests non-stack-bound task group (the group that is allocated by one thread and destroyed by the other)
void TestVagabondGroup () {
    NativeParallelFor( 2, SharedGroupBody(2, VagabondGroup) );
}

//------------------------------------------------------------------------
// Common requisites of the Fibonacci tests
//------------------------------------------------------------------------

const uint_t N = 20;
const uint_t F = 6765;

atomic_t g_Sum;

#define FIB_TEST_PROLOGUE() \
    const unsigned numRepeats = g_MaxConcurrency * (TBB_USE_DEBUG ? 4 : 16);    \
    Harness::ConcurrencyTracker::Reset()

#define FIB_TEST_EPILOGUE(sum) \
    ASSERT( sum == numRepeats * F, NULL ); \
    REMARK("Realized parallelism in Fib test is %u out of %u\n", Harness::ConcurrencyTracker::PeakParallelism(), g_MaxConcurrency)

//------------------------------------------------------------------------
// Test for a complex tree of task groups
//
// The test executes a tree of task groups of the same sort with asymmetric
// descendant nodes distribution at each level at each level.
//
// The chores are specified as functor objects. Each task group contains only one chore.
//------------------------------------------------------------------------

template<uint_t Func(uint_t)>
struct FibTask : NoAssign, Harness::NoAfterlife {
    uint_t* m_pRes;
    const uint_t m_Num;
    FibTask( uint_t* y, uint_t n ) : m_pRes(y), m_Num(n) {}
    void operator() () const {
        *m_pRes = Func(m_Num);
    }
};

uint_t Fib_SpawnRightChildOnly ( uint_t n ) {
    Harness::ConcurrencyTracker ct;
    if( n<2 ) {
        return n;
    } else {
        uint_t y = ~0u;
        Concurrency::task_group tg;
        tg.run( FibTask<Fib_SpawnRightChildOnly>(&y, n-1) );
        uint_t x = Fib_SpawnRightChildOnly(n-2);
        tg.wait();
        return y+x;
    }
}

void TestFib1 () {
    FIB_TEST_PROLOGUE();
    uint_t sum = 0;
    for( unsigned i = 0; i < numRepeats; ++i )
        sum += Fib_SpawnRightChildOnly(N);
    FIB_TEST_EPILOGUE(sum);
}


//------------------------------------------------------------------------
// Test for a mixed tree of task groups.
//
// The test executes a tree with multiple task of one sort at the first level,
// each of which originates in its turn a binary tree of descendant task groups.
//
// The chores are specified both as functor objects and as function pointers
//------------------------------------------------------------------------

uint_t Fib_SpawnBothChildren( uint_t n ) {
    Harness::ConcurrencyTracker ct;
    if( n<2 ) {
        return n;
    } else {
        uint_t  y = ~0u,
                x = ~0u;
        Concurrency::task_group tg;
        tg.run( FibTask<Fib_SpawnBothChildren>(&x, n-2) );
        tg.run( FibTask<Fib_SpawnBothChildren>(&y, n-1) );
        tg.wait();
        return y + x;
    }
}

void RunFib2 () {
    g_Sum += Fib_SpawnBothChildren(N);
}

void TestFib2 () {
    FIB_TEST_PROLOGUE();
    g_Sum = 0;
    Concurrency::task_group rg;
    for( unsigned i = 0; i < numRepeats - 1; ++i )
        rg.run( &RunFib2 );
    rg.wait();
    rg.run( &RunFib2 );
    rg.wait();
    FIB_TEST_EPILOGUE(g_Sum);
}


//------------------------------------------------------------------------
// Test for a complex tree of task groups
// The chores are specified as task handles for recursive functor objects.
//------------------------------------------------------------------------

class FibTask_SpawnRightChildOnly : NoAssign, Harness::NoAfterlife {
    uint_t* m_pRes;
    mutable uint_t m_Num;

public:
    FibTask_SpawnRightChildOnly( uint_t* y, uint_t n ) : m_pRes(y), m_Num(n) {}
    void operator() () const {
        Harness::ConcurrencyTracker ct;
        AssertLive();
        if( m_Num < 2 ) {
            *m_pRes = m_Num;
        } else {
            uint_t y = ~0u;
            Concurrency::task_group tg;
            Concurrency::task_handle<FibTask_SpawnRightChildOnly> h = FibTask_SpawnRightChildOnly(&y, m_Num-1);
            tg.run( h );
            m_Num -= 2;
            tg.run_and_wait( *this );
            *m_pRes += y;
        }
    }
};

uint_t RunFib3 ( uint_t n ) {
    uint_t res = ~0u;
    FibTask_SpawnRightChildOnly func(&res, n);
    func();
    return res;
}

void TestTaskHandle () {
    FIB_TEST_PROLOGUE();
    uint_t sum = 0;
    for( unsigned i = 0; i < numRepeats; ++i )
        sum += RunFib3(N);
    FIB_TEST_EPILOGUE(sum);
}

//------------------------------------------------------------------------
// Test for a mixed tree of task groups.
// The chores are specified as task handles for both functor objects and function pointers
//------------------------------------------------------------------------

template<class task_group_type>
class FibTask_SpawnBothChildren : NoAssign, Harness::NoAfterlife {
    uint_t* m_pRes;
    uint_t m_Num;
public:
    FibTask_SpawnBothChildren( uint_t* y, uint_t n ) : m_pRes(y), m_Num(n) {}
    void operator() () const {
        Harness::ConcurrencyTracker ct;
        AssertLive();
        if( m_Num < 2 ) {
            *m_pRes = m_Num;
        } else {
            uint_t  x = ~0u, // initialized only to suppress warning
                    y = ~0u;
            task_group_type tg;
            Concurrency::task_handle<FibTask_SpawnBothChildren> h1 = FibTask_SpawnBothChildren(&y, m_Num-1),
                                                                h2 = FibTask_SpawnBothChildren(&x, m_Num-2);
            tg.run( h1 );
            tg.run( h2 );
            tg.wait();
            *m_pRes = x + y;
        }
    }
};

template<class task_group_type>
void RunFib4 () {
    uint_t res = ~0u;
    FibTask_SpawnBothChildren<task_group_type> func(&res, N);
    func();
    g_Sum += res;
}

template<class task_group_type>
void TestTaskHandle2 () {
    FIB_TEST_PROLOGUE();
    g_Sum = 0;
    task_group_type rg;
    typedef tbb::aligned_space<handle_type> handle_space_t;
    handle_space_t *handles = new handle_space_t[numRepeats];
    handle_type *h = NULL;
#if __TBB_ipf && __TBB_GCC_VERSION==40601
    volatile // Workaround for unexpected exit from the loop below after the exception was caught
#endif
    unsigned i = 0;
    for( ;; ++i ) {
        h = handles[i].begin();
#if __TBB_FUNC_PTR_AS_TEMPL_PARAM_BROKEN
        new ( h ) handle_type((void(*)())RunFib4<task_group_type>);
#else
        new ( h ) handle_type(RunFib4<task_group_type>);
#endif
        if ( i == numRepeats - 1 )
            break;
        rg.run( *h );
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
        bool caught = false;
        try {
            if( i&1 ) rg.run( *h );
            else rg.run_and_wait( *h );
        }
        catch ( Concurrency::invalid_multiple_scheduling& e ) {
            ASSERT( e.what(), "Error message is absent" );
            caught = true;
        }
        catch ( ... ) {
            ASSERT ( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unrecognized exception" );
        }
        ASSERT ( caught, "Expected invalid_multiple_scheduling exception is missing" );
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
    }
    ASSERT( i == numRepeats - 1, "unexpected exit from the loop" );
    rg.run_and_wait( *h );

    for( i = 0; i < numRepeats; ++i )
#if __TBB_UNQUALIFIED_CALL_OF_DTOR_BROKEN
        handles[i].begin()->Concurrency::task_handle<void(*)()>::~task_handle();
#else
        handles[i].begin()->~handle_type();
#endif
    delete []handles;
    FIB_TEST_EPILOGUE(g_Sum);
}

#if __TBB_CPP11_LAMBDAS_PRESENT
//------------------------------------------------------------------------
// Test for a mixed tree of task groups.
// The chores are specified as lambdas
//------------------------------------------------------------------------

void TestFibWithLambdas () {
    REMARK ("Lambdas test");
    FIB_TEST_PROLOGUE();
    atomic_t sum;
    sum = 0;
    Concurrency::task_group rg;
    for( unsigned i = 0; i < numRepeats; ++i )
        rg.run( [&](){sum += Fib_SpawnBothChildren(N);} );
    rg.wait();
    FIB_TEST_EPILOGUE(sum);
}

//------------------------------------------------------------------------
// Test for make_task.
// The chores are specified as lambdas converted to task_handles.
//------------------------------------------------------------------------

void TestFibWithMakeTask () {
    REMARK ("Make_task test\n");
    atomic_t sum;
    sum = 0;
    Concurrency::task_group rg;
    const auto &h1 = Concurrency::make_task( [&](){sum += Fib_SpawnBothChildren(N);} );
    const auto &h2 = Concurrency::make_task( [&](){sum += Fib_SpawnBothChildren(N);} );
    rg.run( h1 );
    rg.run_and_wait( h2 );
    ASSERT( sum == 2 * F, NULL );
}
#endif /* __TBB_CPP11_LAMBDAS_PRESENT */


//------------------------------------------------------------------------
// Tests for exception handling and cancellation behavior.
//------------------------------------------------------------------------

class test_exception : public std::exception
{
    const char* m_strDescription;
public:
    test_exception ( const char* descr ) : m_strDescription(descr) {}

    const char* what() const throw() __TBB_override { return m_strDescription; }
};

#if TBB_USE_CAPTURED_EXCEPTION
    #include "tbb/tbb_exception.h"
    typedef tbb::captured_exception TestException;
#else
    typedef test_exception TestException;
#endif

#include <string.h>

#define NUM_CHORES      512
#define NUM_GROUPS      64
#define SKIP_CHORES     (NUM_CHORES/4)
#define SKIP_GROUPS     (NUM_GROUPS/4)
#define EXCEPTION_DESCR1 "Test exception 1"
#define EXCEPTION_DESCR2 "Test exception 2"

atomic_t g_ExceptionCount;
atomic_t g_TaskCount;
unsigned g_ExecutedAtCancellation;
bool g_Rethrow;
bool g_Throw;
#if __TBB_SILENT_CANCELLATION_BROKEN
    volatile bool g_CancellationPropagationInProgress;
    #define CATCH_ANY()                                     \
        __TBB_CATCH( ... ) {                                \
            if ( g_CancellationPropagationInProgress ) {    \
                if ( g_Throw ) {                            \
                    exceptionCaught = true;                 \
                    ++g_ExceptionCount;                     \
                }                                           \
            } else                                          \
                ASSERT( false, "Unknown exception" );       \
        }
#else
    #define CATCH_ANY()  __TBB_CATCH( ... ) { ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unknown exception" ); }
#endif

inline
void ResetGlobals ( bool bThrow, bool bRethrow ) {
    g_Throw = bThrow;
    g_Rethrow = bRethrow;
#if __TBB_SILENT_CANCELLATION_BROKEN
    g_CancellationPropagationInProgress = false;
#endif
    g_ExceptionCount = 0;
    g_TaskCount = 0;
    Harness::ConcurrencyTracker::Reset();
}

class ThrowingTask : NoAssign, Harness::NoAfterlife {
    atomic_t &m_TaskCount;
public:
    ThrowingTask( atomic_t& counter ) : m_TaskCount(counter) {}
    void operator() () const {
        Harness::ConcurrencyTracker ct;
        AssertLive();
        if ( g_Throw ) {
            if ( ++m_TaskCount == SKIP_CHORES )
                __TBB_THROW( test_exception(EXCEPTION_DESCR1) );
            __TBB_Yield();
        }
        else {
            ++g_TaskCount;
            while( !Concurrency::is_current_task_group_canceling() )
                __TBB_Yield();
        }
    }
};

void LaunchChildren () {
    atomic_t count;
    count = 0;
    Concurrency::task_group g;
    bool exceptionCaught = false;
    for( unsigned i = 0; i < NUM_CHORES; ++i )
        g.run( ThrowingTask(count) );
    Concurrency::task_group_status status = Concurrency::not_complete;
    __TBB_TRY {
        status = g.wait();
    } __TBB_CATCH ( TestException& e ) {
#if TBB_USE_EXCEPTIONS
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR1) == 0, "Unknown exception" );
#endif /* TBB_USE_EXCEPTIONS */
        exceptionCaught = true;
        ++g_ExceptionCount;
    } CATCH_ANY();
    ASSERT( !g_Throw || exceptionCaught || status == Concurrency::canceled, "No exception in the child task group" );
    if ( g_Rethrow && g_ExceptionCount > SKIP_GROUPS ) {
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        __TBB_THROW( test_exception(EXCEPTION_DESCR2) );
    }
}

#if TBB_USE_EXCEPTIONS
void TestEh1 () {
    ResetGlobals( true, false );
    Concurrency::task_group rg;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        rg.run( &LaunchChildren );
    try {
        rg.wait();
    } catch ( ... ) {
        ASSERT( false, "Unexpected exception" );
    }
    ASSERT( g_ExceptionCount <= NUM_GROUPS, "Too many exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount == NUM_GROUPS, "Not all child groups threw the exception" );
}

void TestEh2 () {
    ResetGlobals( true, true );
    Concurrency::task_group rg;
    bool exceptionCaught = false;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        rg.run( &LaunchChildren );
    try {
        rg.wait();
    } catch ( TestException& e ) {
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR2) == 0, "Unknown exception" );
        ASSERT ( !rg.is_canceling(), "wait() has not reset cancellation state" );
        exceptionCaught = true;
    } CATCH_ANY();
    ASSERT( exceptionCaught, "No exception thrown from the root task group" );
    ASSERT( g_ExceptionCount >= SKIP_GROUPS, "Too few exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount <= NUM_GROUPS - SKIP_GROUPS, "Too many exceptions from the child groups. The test is broken" );
    ASSERT( g_ExceptionCount < NUM_GROUPS - SKIP_GROUPS, "None of the child groups was cancelled" );
}
#endif /* TBB_USE_EXCEPTIONS */

//------------------------------------------------------------------------
// Tests for manual cancellation of the task_group hierarchy
//------------------------------------------------------------------------

void TestCancellation1 () {
    ResetGlobals( false, false );
    Concurrency::task_group rg;
    for( unsigned i = 0; i < NUM_GROUPS; ++i )
        // TBB version does not require taking function address
        rg.run( &LaunchChildren );
    ASSERT ( !Concurrency::is_current_task_group_canceling(), "Unexpected cancellation" );
    ASSERT ( !rg.is_canceling(), "Unexpected cancellation" );
#if __TBB_SILENT_CANCELLATION_BROKEN
    g_CancellationPropagationInProgress = true;
#endif
    while ( g_MaxConcurrency > 1 && g_TaskCount == 0 )
        __TBB_Yield();
    rg.cancel();
    g_ExecutedAtCancellation = g_TaskCount;
    ASSERT ( rg.is_canceling(), "No cancellation reported" );
    rg.wait();
    ASSERT( g_TaskCount <= NUM_GROUPS * NUM_CHORES, "Too many tasks reported. The test is broken" );
    ASSERT( g_TaskCount < NUM_GROUPS * NUM_CHORES, "No tasks were cancelled. Cancellation model changed?" );
    ASSERT( g_TaskCount <= g_ExecutedAtCancellation + Harness::ConcurrencyTracker::PeakParallelism(), "Too many tasks survived cancellation" );
}

//------------------------------------------------------------------------
// Tests for manual cancellation of the structured_task_group hierarchy
//------------------------------------------------------------------------

void StructuredLaunchChildren () {
    atomic_t count;
    count = 0;
    Concurrency::structured_task_group g;
    bool exceptionCaught = false;
    typedef Concurrency::task_handle<ThrowingTask> throwing_handle_type;
    tbb::aligned_space<throwing_handle_type,NUM_CHORES> handles;
    for( unsigned i = 0; i < NUM_CHORES; ++i ) {
        throwing_handle_type *h = handles.begin()+i;
        new ( h ) throwing_handle_type( ThrowingTask(count) );
        g.run( *h );
    }
    __TBB_TRY {
        g.wait();
    } __TBB_CATCH( TestException& e ) {
#if TBB_USE_EXCEPTIONS
        ASSERT( e.what(), "Empty what() string" );
        ASSERT( __TBB_EXCEPTION_TYPE_INFO_BROKEN || strcmp(e.what(), EXCEPTION_DESCR1) == 0, "Unknown exception" );
#endif /* TBB_USE_EXCEPTIONS */
#if __TBB_SILENT_CANCELLATION_BROKEN
        ASSERT ( !g.is_canceling() || g_CancellationPropagationInProgress, "wait() has not reset cancellation state" );
#else
        ASSERT ( !g.is_canceling(), "wait() has not reset cancellation state" );
#endif
        exceptionCaught = true;
        ++g_ExceptionCount;
    } CATCH_ANY();
    ASSERT( !g_Throw || exceptionCaught, "No exception in the child task group" );
    for( unsigned i = 0; i < NUM_CHORES; ++i )
        (handles.begin()+i)->~throwing_handle_type();
    if ( g_Rethrow && g_ExceptionCount > SKIP_GROUPS ) {
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        __TBB_THROW( test_exception(EXCEPTION_DESCR2) );
    }
}

class StructuredCancellationTestDriver {
    tbb::aligned_space<handle_type,NUM_CHORES> m_handles;

public:
    void Launch ( Concurrency::structured_task_group& rg ) {
        ResetGlobals( false, false );
        for( unsigned i = 0; i < NUM_GROUPS; ++i ) {
            handle_type *h = m_handles.begin()+i;
            new ( h ) handle_type( StructuredLaunchChildren );
            rg.run( *h );
        }
        ASSERT ( !Concurrency::is_current_task_group_canceling(), "Unexpected cancellation" );
        ASSERT ( !rg.is_canceling(), "Unexpected cancellation" );
#if __TBB_SILENT_CANCELLATION_BROKEN
        g_CancellationPropagationInProgress = true;
#endif
        while ( g_MaxConcurrency > 1 && g_TaskCount == 0 )
            __TBB_Yield();
    }

    void Finish () {
        for( unsigned i = 0; i < NUM_GROUPS; ++i )
            (m_handles.begin()+i)->~handle_type();
        ASSERT( g_TaskCount <= NUM_GROUPS * NUM_CHORES, "Too many tasks reported. The test is broken" );
        ASSERT( g_TaskCount < NUM_GROUPS * NUM_CHORES, "No tasks were cancelled. Cancellation model changed?" );
        ASSERT( g_TaskCount <= g_ExecutedAtCancellation + g_MaxConcurrency, "Too many tasks survived cancellation" );
    }
}; // StructuredCancellationTestDriver

void TestStructuredCancellation1 () {
    StructuredCancellationTestDriver driver;
    Concurrency::structured_task_group sg;
    driver.Launch( sg );
    sg.cancel();
    g_ExecutedAtCancellation = g_TaskCount;
    ASSERT ( sg.is_canceling(), "No cancellation reported" );
    sg.wait();
    driver.Finish();
}

#if TBB_USE_EXCEPTIONS
#if defined(_MSC_VER)
    #pragma warning (disable: 4127)
#endif

template<bool Throw>
void TestStructuredCancellation2 () {
    bool exception_occurred = false,
         unexpected_exception = false;
    StructuredCancellationTestDriver driver;
    try {
        Concurrency::structured_task_group tg;
        driver.Launch( tg );
        if ( Throw )
            throw int(); // Initiate stack unwinding
    }
    catch ( const Concurrency::missing_wait& e ) {
        ASSERT( e.what(), "Error message is absent" );
        exception_occurred = true;
        unexpected_exception = Throw;
    }
    catch ( int ) {
        exception_occurred = true;
        unexpected_exception = !Throw;
    }
    catch ( ... ) {
        exception_occurred = unexpected_exception = true;
    }
    ASSERT( exception_occurred, NULL );
    ASSERT( !unexpected_exception, NULL );
    driver.Finish();
}
#endif /* TBB_USE_EXCEPTIONS */

void EmptyFunction () {}

void TestStructuredWait () {
    Concurrency::structured_task_group sg;
    handle_type h(EmptyFunction);
    sg.run(h);
    sg.wait();
    handle_type h2(EmptyFunction);
    sg.run(h2);
    sg.wait();
}

struct TestFunctor {
    void operator()() { ASSERT( false, "Non-const operator called" ); }
    void operator()() const { /* library requires this overload only */ }
};

void TestConstantFunctorRequirement() {
    tbb::task_group g;
    TestFunctor tf;
    g.run( tf ); g.wait();
    g.run_and_wait( tf );
}
//------------------------------------------------------------------------
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
        MovePreferableFunctor(MovePreferableFunctor&& other) : Movable(std::move(other)) {};
        MovePreferableFunctor(const MovePreferableFunctor& other) : Movable(other) {};
    };

    struct NoMoveNoCopyFunctor : NoCopy, TestFunctor {
        NoMoveNoCopyFunctor() : NoCopy() {};
        // mv ctor is not allowed as cp ctor from parent NoCopy
    private:
        NoMoveNoCopyFunctor(NoMoveNoCopyFunctor&&);
    };

    void TestFunctorsWithinTaskHandles() {
        // working with task_handle rvalues is not supported in task_group

        tbb::task_group tg;
        MovePreferableFunctor mpf;
        typedef tbb::task_handle<MoveOnlyFunctor> th_mv_only_type;
        typedef tbb::task_handle<MovePreferableFunctor> th_mv_pref_type;

        th_mv_only_type th_mv_only = th_mv_only_type(MoveOnlyFunctor());
        tg.run_and_wait(th_mv_only);

        th_mv_only_type th_mv_only1 = th_mv_only_type(MoveOnlyFunctor());
        tg.run(th_mv_only1);
        tg.wait();

        th_mv_pref_type th_mv_pref = th_mv_pref_type(mpf);
        tg.run_and_wait(th_mv_pref);
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref1 = th_mv_pref_type(std::move(mpf));
        tg.run_and_wait(th_mv_pref1);
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref2 = th_mv_pref_type(mpf);
        tg.run(th_mv_pref2);
        tg.wait();
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        th_mv_pref_type th_mv_pref3 = th_mv_pref_type(std::move(mpf));
        tg.run(th_mv_pref3);
        tg.wait();
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }

    void TestBareFunctors() {
        tbb::task_group tg;
        MovePreferableFunctor mpf;
        // run_and_wait() doesn't have any copies or moves of arguments inside the impl
        tg.run_and_wait( NoMoveNoCopyFunctor() );

        tg.run( MoveOnlyFunctor() );
        tg.wait();

        tg.run( mpf );
        tg.wait();
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        tg.run( std::move(mpf) );
        tg.wait();
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }

    void TestMakeTask() {
        MovePreferableFunctor mpf;

        tbb::make_task( MoveOnly() );

        tbb::make_task( mpf );
        ASSERT(mpf.alive, "object was moved when was passed by lval");
        mpf.Reset();

        tbb::make_task( std::move(mpf) );
        ASSERT(!mpf.alive, "object was copied when was passed by rval");
        mpf.Reset();
    }
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

void TestMoveSemantics() {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveSemanticsNS::TestBareFunctors();
    TestMoveSemanticsNS::TestFunctorsWithinTaskHandles();
    TestMoveSemanticsNS::TestMakeTask();
#else
    REPORT("Known issue: move support tests are skipped.\n");
#endif
}
//------------------------------------------------------------------------


int TestMain () {
    REMARK ("Testing %s task_group functionality\n", TBBTEST_USE_TBB ? "TBB" : "PPL");
    for( int p=MinThread; p<=MaxThread; ++p ) {
        g_MaxConcurrency = p;
#if TBBTEST_USE_TBB
        tbb::task_scheduler_init init(p);
#else
        Concurrency::SchedulerPolicy sp( 4,
                                Concurrency::SchedulerKind, Concurrency::ThreadScheduler,
                                Concurrency::MinConcurrency, 1,
                                Concurrency::MaxConcurrency, p,
                                Concurrency::TargetOversubscriptionFactor, 1);
        Concurrency::Scheduler  *s = Concurrency::Scheduler::Create( sp );
#endif /* !TBBTEST_USE_TBB */
        if ( p > 1 ) {
            TestParallelSpawn();
            TestParallelWait();
            TestVagabondGroup();
        }
        TestFib1();
        TestFib2();
        TestTaskHandle();
        TestTaskHandle2<Concurrency::task_group>();
        TestTaskHandle2<Concurrency::structured_task_group>();
#if __TBB_CPP11_LAMBDAS_PRESENT
        TestFibWithLambdas();
        TestFibWithMakeTask();
#endif
        TestCancellation1();
        TestStructuredCancellation1();
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
        TestEh1();
        TestEh2();
        TestStructuredWait();
        TestStructuredCancellation2<true>();
#if !(__TBB_THROW_FROM_DTOR_BROKEN || __TBB_STD_UNCAUGHT_EXCEPTION_BROKEN)
        TestStructuredCancellation2<false>();
#else
        REPORT("Known issue: TestStructuredCancellation2<false>() is skipped.\n");
#endif
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
#if !TBBTEST_USE_TBB
        s->Release();
#endif
    }
    TestConstantFunctorRequirement();
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception handling tests are skipped.\n");
#endif
    TestMoveSemantics();
    return Harness::Done;
}

#else /* !__TBB_TASK_GROUP_CONTEXT */

#include "harness.h"

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_TASK_GROUP_CONTEXT */
