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

#define HARNESS_DEFAULT_MIN_THREADS 2
#define HARNESS_DEFAULT_MAX_THREADS 4

#include "harness.h"

#if __TBB_TASK_GROUP_CONTEXT

#include <limits.h> // for INT_MAX
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb_exception.h"
#include "tbb/task.h"
#include "tbb/atomic.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_do.h"
#include "tbb/pipeline.h"
#include "tbb/parallel_scan.h"
#include "tbb/blocked_range.h"
#include "harness_assert.h"

#define FLAT_RANGE  100000
#define FLAT_GRAIN  100
#define OUTER_RANGE  100
#define OUTER_GRAIN  10
#define INNER_RANGE  (FLAT_RANGE / OUTER_RANGE)
#define INNER_GRAIN  (FLAT_GRAIN / OUTER_GRAIN)

tbb::atomic<intptr_t> g_FedTasksCount; // number of tasks added by parallel_do feeder
tbb::atomic<intptr_t> g_OuterParCalls;  // number of actual invocations of the outer construct executed.
tbb::atomic<intptr_t> g_TGCCancelled;  // Number of times a task sees its group cancelled at start

inline intptr_t Existed () { return INT_MAX; }

#include "harness_eh.h"
/********************************
      Variables in test

__ Test control variables
      g_ExceptionInMaster -- only the master thread is allowed to throw.  If false, the master cannot throw
      g_SolitaryException -- only one throw may be executed.

-- controls for ThrowTestException for pipeline tests
      g_NestedPipelines -- are inner pipelines being run?
      g_PipelinesStarted -- how many pipelines have run their first filter at least once.

-- Information variables

   g_Master -- Thread ID of the "master" thread
      In pipelines sometimes the master thread does not participate, so the tests have to be resilient to this.

-- Measurement variables

   g_OuterParCalls -- how many outer parallel ranges or filters started
   g_TGCCancelled --  how many inner parallel ranges or filters saw task::self().is_cancelled()
   g_ExceptionsThrown -- number of throws executed (counted in ThrowTestException)
   g_MasterExecutedThrow -- number of times master thread actually executed a throw
   g_NonMasterExecutedThrow -- number of times non-master thread actually executed a throw
   g_ExceptionCaught -- one of PropagatedException or unknown exception was caught.  (Other exceptions cause assertions.)

   --  Tallies for the task bodies which have executed (counted in each inner body, sampled in ThrowTestException)
      g_CurExecuted -- total number of inner ranges or filters which executed
      g_ExecutedAtLastCatch -- value of g_CurExecuted when last catch was made, 0 if none.
      g_ExecutedAtFirstCatch -- value of g_CurExecuted when first catch is made, 0 if none.
  *********************************/

inline void ResetGlobals (  bool throwException = true, bool flog = false ) {
    ResetEhGlobals( throwException, flog );
    g_FedTasksCount = 0;
    g_OuterParCalls = 0;
    g_NestedPipelines = false;
    g_TGCCancelled = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Tests for tbb::parallel_for and tbb::parallel_reduce

typedef size_t count_type;
typedef tbb::blocked_range<count_type> range_type;

inline intptr_t CountSubranges(range_type r) {
    if(!r.is_divisible()) return intptr_t(1);
    range_type r2(r,tbb::split());
    return CountSubranges(r) + CountSubranges(r2);
}

inline intptr_t NumSubranges ( intptr_t length, intptr_t grain ) {
    return CountSubranges(range_type(0,length,grain));
}

template<class Body>
intptr_t TestNumSubrangesCalculation ( intptr_t length, intptr_t grain, intptr_t inner_length, intptr_t inner_grain ) {
    ResetGlobals();
    g_ThrowException = false;
    intptr_t outerCalls = NumSubranges(length, grain),
             innerCalls = NumSubranges(inner_length, inner_grain),
             maxExecuted = outerCalls * (innerCalls + 1);
    tbb::parallel_for( range_type(0, length, grain), Body() );
    ASSERT (g_CurExecuted == maxExecuted, "Wrong estimation of bodies invocation count");
    return maxExecuted;
}

class NoThrowParForBody {
public:
    void operator()( const range_type& r ) const {
        volatile count_type x = 0;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        count_type end = r.end();
        for( count_type i=r.begin(); i<end; ++i )
            x += i;
    }
};

#if TBB_USE_EXCEPTIONS

void Test0 () {
    ResetGlobals();
    tbb::simple_partitioner p;
    for( size_t i=0; i<10; ++i ) {
        tbb::parallel_for( range_type(0, 0, 1), NoThrowParForBody() );
        tbb::parallel_for( range_type(0, 0, 1), NoThrowParForBody(), p );
        tbb::parallel_for( range_type(0, 128, 8), NoThrowParForBody() );
        tbb::parallel_for( range_type(0, 128, 8), NoThrowParForBody(), p );
    }
} // void Test0 ()

//! Template that creates a functor suitable for parallel_reduce from a functor for parallel_for.
template<typename ParForBody>
class SimpleParReduceBody: NoAssign {
    ParForBody m_Body;
public:
    void operator()( const range_type& r ) const { m_Body(r); }
    SimpleParReduceBody() {}
    SimpleParReduceBody( SimpleParReduceBody& left, tbb::split ) : m_Body(left.m_Body) {}
    void join( SimpleParReduceBody& /*right*/ ) {}
}; // SimpleParReduceBody

//! Test parallel_for and parallel_reduce for a given partitioner.
/** The Body need only be suitable for a parallel_for. */
template<typename ParForBody, typename Partitioner>
void TestParallelLoopAux() {
    Partitioner partitioner;
    for( int i=0; i<2; ++i ) {
        ResetGlobals();
        TRY();
            if( i==0 )
                tbb::parallel_for( range_type(0, FLAT_RANGE, FLAT_GRAIN), ParForBody(), partitioner );
            else {
                SimpleParReduceBody<ParForBody> rb;
                tbb::parallel_reduce( range_type(0, FLAT_RANGE, FLAT_GRAIN), rb, partitioner );
            }
        CATCH_AND_ASSERT();
        // two cases: g_SolitaryException and !g_SolitaryException
        //   1) g_SolitaryException: only one thread actually threw.  There is only one context, so the exception
        //      (when caught) will cause that context to be cancelled.  After this event, there may be one or
        //      more threads which are "in-flight", up to g_NumThreads, but no more will be started.  The threads,
        //      when they start, if they see they are cancelled, TGCCancelled is incremented.
        //   2) !g_SolitaryException: more than one thread can throw.  The number of threads that actually
        //      threw is g_MasterExecutedThrow if only the master is allowed, else g_NonMasterExecutedThrow.
        //      Only one context, so TGCCancelled should be <= g_NumThreads.
        //
        // the reasoning is similar for nested algorithms in a single context (Test2).
        //
        // If a thread throws in a context, more than one subsequent task body may see the
        // cancelled state (if they are scheduled before the state is propagated.) this is
        // infrequent, but it occurs.  So what was to be an assertion must be a remark.
        ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks ran after exception thrown");
        if( g_TGCCancelled > g_NumThreads) REMARK( "Too many tasks ran after exception thrown (%d vs. %d)\n",
                (int)g_TGCCancelled, (int)g_NumThreads);
        ASSERT(g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
        if ( g_SolitaryException ) {
            ASSERT(g_NumExceptionsCaught == 1, "No try_blocks in any body expected in this test");
            ASSERT(g_NumExceptionsCaught == (g_ExceptionInMaster ? g_MasterExecutedThrow : g_NonMasterExecutedThrow),
                "Not all throws were caught");
            ASSERT(g_ExecutedAtFirstCatch == g_ExecutedAtLastCatch, "Too many exceptions occurred");
        }
        else {
            ASSERT(g_NumExceptionsCaught >= 1, "No try blocks in any body expected in this test");
        }
    }
}  // TestParallelLoopAux

//! Test with parallel_for and parallel_reduce, over all three kinds of partitioners.
/** The Body only needs to be suitable for tbb::parallel_for. */
template<typename Body>
void TestParallelLoop() {
    // The simple and auto partitioners should be const, but not the affinity partitioner.
    TestParallelLoopAux<Body, const tbb::simple_partitioner  >();
    TestParallelLoopAux<Body, const tbb::auto_partitioner    >();
#define __TBB_TEMPORARILY_DISABLED 1
#if !__TBB_TEMPORARILY_DISABLED
    // TODO: Improve the test so that it tolerates delayed start of tasks with affinity_partitioner
    TestParallelLoopAux<Body, /***/ tbb::affinity_partitioner>();
#endif
#undef __TBB_TEMPORARILY_DISABLED
}

class SimpleParForBody: NoAssign {
public:
    void operator()( const range_type& r ) const {
        Harness::ConcurrencyTracker ct;
        volatile long x = 0;
        ++g_CurExecuted;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        for( count_type i = r.begin(); i != r.end(); ++i )
            x += 0;
        WaitUntilConcurrencyPeaks();
        ThrowTestException(1);
    }
};

void Test1() {
    // non-nested parallel_for/reduce with throwing body, one context
    TestParallelLoop<SimpleParForBody>();
} // void Test1 ()

class OuterParForBody: NoAssign {
public:
    void operator()( const range_type& ) const {
        Harness::ConcurrencyTracker ct;
        ++g_OuterParCalls;
        tbb::parallel_for( tbb::blocked_range<size_t>(0, INNER_RANGE, INNER_GRAIN), SimpleParForBody() );
    }
};

//! Uses parallel_for body containing an inner parallel_for with the default context not wrapped by a try-block.
/** Inner algorithms are spawned inside the new bound context by default. Since
    exceptions thrown from the inner parallel_for are not handled by the caller
    (outer parallel_for body) in this test, they will cancel all the sibling inner
    algorithms. **/
void Test2 () {
    TestParallelLoop<OuterParForBody>();
} // void Test2 ()

class OuterParForBodyWithIsolatedCtx {
public:
    void operator()( const range_type& ) const {
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        ++g_OuterParCalls;
        tbb::parallel_for( tbb::blocked_range<size_t>(0, INNER_RANGE, INNER_GRAIN), SimpleParForBody(), tbb::simple_partitioner(), ctx );
    }
};

//! Uses parallel_for body invoking an inner parallel_for with an isolated context without a try-block.
/** Even though exceptions thrown from the inner parallel_for are not handled
    by the caller in this test, they will not affect sibling inner algorithms
    already running because of the isolated contexts. However because the first
    exception cancels the root parallel_for only the first g_NumThreads subranges
    will be processed (which launch inner parallel_fors) **/
void Test3 () {
    ResetGlobals();
    typedef OuterParForBodyWithIsolatedCtx body_type;
    intptr_t  innerCalls = NumSubranges(INNER_RANGE, INNER_GRAIN),
            // we expect one thread to throw without counting, the rest to run to completion
            // this formula assumes g_numThreads outer pfor ranges will be started, but that is not the
            // case; the SimpleParFor subranges are started up as part of the outer ones, and when
            // the amount of concurrency reaches g_NumThreads no more outer Pfor ranges are started.
            // so we have to count the number of outer Pfors actually started.
            minExecuted = (g_NumThreads - 1) * innerCalls;
    TRY();
        tbb::parallel_for( range_type(0, OUTER_RANGE, OUTER_GRAIN), body_type() );
    CATCH_AND_ASSERT();
    minExecuted = (g_OuterParCalls - 1) * innerCalls;  // see above

    // The first formula above assumes all ranges of the outer parallel for are executed, and one
    // cancels.  In the event, we have a smaller number of ranges that start before the exception
    // is caught.
    //
    //  g_SolitaryException:One inner range throws.  Outer parallel_For is cancelled, but sibling
    //                      parallel_fors continue to completion (unless the threads that execute
    //                      are not allowed to throw, in which case we will not see any exceptions).
    // !g_SolitaryException:multiple inner ranges may throw.  Any which throws will stop, and the
    //                      corresponding range of the outer pfor will stop also.
    //
    // In either case, once the outer pfor gets the exception it will stop executing further ranges.

    // if the only threads executing were not allowed to throw, then not seeing an exception is okay.
    bool okayNoExceptionsCaught = (g_ExceptionInMaster && !g_MasterExecuted) || (!g_ExceptionInMaster && !g_NonMasterExecuted);
    if ( g_SolitaryException ) {
        ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");
        ASSERT (g_CurExecuted > minExecuted, "Too few tasks survived exception");
        ASSERT (g_CurExecuted <= minExecuted + (g_ExecutedAtLastCatch + g_NumThreads), "Too many tasks survived exception");
        ASSERT (g_NumExceptionsCaught == 1 || okayNoExceptionsCaught, "No try_blocks in any body expected in this test");
    }
    else {
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
        ASSERT (g_NumExceptionsCaught >= 1 || okayNoExceptionsCaught, "No try_blocks in any body expected in this test");
    }
} // void Test3 ()

class OuterParForExceptionSafeBody {
public:
    void operator()( const range_type& ) const {
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        ++g_OuterParCalls;
        TRY();
            tbb::parallel_for( tbb::blocked_range<size_t>(0, INNER_RANGE, INNER_GRAIN), SimpleParForBody(), tbb::simple_partitioner(), ctx );
        CATCH();  // this macro sets g_ExceptionCaught
    }
};

//! Uses parallel_for body invoking an inner parallel_for (with isolated context) inside a try-block.
/** Since exception(s) thrown from the inner parallel_for are handled by the caller
    in this test, they do not affect neither other tasks of the the root parallel_for
    nor sibling inner algorithms. **/
void Test4 () {
    ResetGlobals( true, true );
    intptr_t  innerCalls = NumSubranges(INNER_RANGE, INNER_GRAIN),
            outerCalls = NumSubranges(OUTER_RANGE, OUTER_GRAIN);
    TRY();
        tbb::parallel_for( range_type(0, OUTER_RANGE, OUTER_GRAIN), OuterParForExceptionSafeBody() );
    CATCH();
    // g_SolitaryException  : one inner pfor will throw, the rest will execute to completion.
    //                        so the count should be (outerCalls -1) * innerCalls, if a throw happened.
    // !g_SolitaryException : possible multiple inner pfor throws.  Should be approximately
    //                        (outerCalls - g_NumExceptionsCaught) * innerCalls, give or take a few
    intptr_t  minExecuted = (outerCalls - g_NumExceptionsCaught) * innerCalls;
    bool okayNoExceptionsCaught = (g_ExceptionInMaster && !g_MasterExecuted) || (!g_ExceptionInMaster && !g_NonMasterExecuted);
    if ( g_SolitaryException ) {
        // only one task had exception thrown. That task had at least one execution (the one that threw).
        // There may be an arbitrary number of ranges executed after the throw but before the exception
        // is caught in the scheduler and cancellation is signaled.  (seen 9, 11 and 62 (!) for 8 threads)
        ASSERT (g_NumExceptionsCaught == 1 || okayNoExceptionsCaught, "No exception registered");
        ASSERT (g_CurExecuted >= minExecuted, "Too few tasks executed");
        ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");
        // a small number of threads can execute in a throwing sub-pfor, if the task which is
        // to do the solitary throw swaps out after registering its intent to throw but before it
        // actually does so.  (Or is this caused by the extra threads participating? No, the
        // number of extra tasks is sometimes far greater than the number of extra threads.)
        ASSERT (g_CurExecuted <= minExecuted + g_NumThreads, "Too many tasks survived exception");
        if(g_CurExecuted > minExecuted + g_NumThreads) REMARK("Unusual number of tasks executed after signal (%d vs. %d)\n",
                (int)g_CurExecuted, minExecuted + g_NumThreads);
    }
    else {
        ASSERT ((g_NumExceptionsCaught >= 1 && g_NumExceptionsCaught <= outerCalls) || okayNoExceptionsCaught, "Unexpected actual number of exceptions");
        ASSERT (g_CurExecuted >= minExecuted, "Too few executed tasks reported");
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived multiple exceptions");
        if(g_CurExecuted > g_ExecutedAtLastCatch + g_NumThreads) REMARK("Unusual number of tasks executed after signal (%d vs. %d)\n",
                (int)g_CurExecuted, g_ExecutedAtLastCatch + g_NumThreads);
        ASSERT (g_CurExecuted <= outerCalls * (1 + g_NumThreads), "Too many tasks survived exception");
    }
} // void Test4 ()

#endif /* TBB_USE_EXCEPTIONS */

class ParForBodyToCancel {
public:
    void operator()( const range_type& ) const {
        ++g_CurExecuted;
        CancellatorTask::WaitUntilReady();
    }
};

template<class B>
class ParForLauncherTask : public tbb::task {
    tbb::task_group_context &my_ctx;

    tbb::task* execute () __TBB_override {
        tbb::parallel_for( range_type(0, FLAT_RANGE, FLAT_GRAIN), B(), tbb::simple_partitioner(), my_ctx );
        return NULL;
    }
public:
    ParForLauncherTask ( tbb::task_group_context& ctx ) : my_ctx(ctx) {}
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
void TestCancelation1 () {
    ResetGlobals( false );
    RunCancellationTest<ParForLauncherTask<ParForBodyToCancel>, CancellatorTask>( NumSubranges(FLAT_RANGE, FLAT_GRAIN) / 4 );
}

class CancellatorTask2 : public tbb::task {
    tbb::task_group_context &m_GroupToCancel;

    tbb::task* execute () __TBB_override {
        Harness::ConcurrencyTracker ct;
        WaitUntilConcurrencyPeaks();
        m_GroupToCancel.cancel_group_execution();
        g_ExecutedAtLastCatch = g_CurExecuted;
        return NULL;
    }
public:
    CancellatorTask2 ( tbb::task_group_context& ctx, intptr_t ) : m_GroupToCancel(ctx) {}
};

class ParForBodyToCancel2 {
public:
    void operator()( const range_type& ) const {
        ++g_CurExecuted;
        Harness::ConcurrencyTracker ct;
        // The test will hang (and be timed out by the test system) if is_cancelled() is broken
        while( !tbb::task::self().is_cancelled() )
            __TBB_Yield();
    }
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
/** This version also tests task::is_cancelled() method. **/
void TestCancelation2 () {
    ResetGlobals();
    RunCancellationTest<ParForLauncherTask<ParForBodyToCancel2>, CancellatorTask2>();
    ASSERT (g_ExecutedAtLastCatch < g_NumThreads, "Somehow worker tasks started their execution before the cancellator task");
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived cancellation");
    ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Some tasks were executed after cancellation");
}

////////////////////////////////////////////////////////////////////////////////
// Regression test based on the contribution by the author of the following forum post:
// http://softwarecommunity.intel.com/isn/Community/en-US/forums/thread/30254959.aspx

class Worker {
    static const int max_nesting = 3;
    static const int reduce_range = 1024;
    static const int reduce_grain = 256;
public:
    int DoWork (int level);
    int Validate (int start_level) {
        int expected = 1; // identity for multiplication
        for(int i=start_level+1; i<max_nesting; ++i)
             expected *= reduce_range;
        return expected;
    }
};

class RecursiveParReduceBodyWithSharedWorker {
    Worker * m_SharedWorker;
    int m_NestingLevel;
    int m_Result;
public:
    RecursiveParReduceBodyWithSharedWorker ( RecursiveParReduceBodyWithSharedWorker& src, tbb::split )
        : m_SharedWorker(src.m_SharedWorker)
        , m_NestingLevel(src.m_NestingLevel)
        , m_Result(0)
    {}
    RecursiveParReduceBodyWithSharedWorker ( Worker *w, int outer )
        : m_SharedWorker(w)
        , m_NestingLevel(outer)
        , m_Result(0)
    {}

    void operator() ( const tbb::blocked_range<size_t>& r ) {
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        for (size_t i = r.begin (); i != r.end (); ++i) {
            m_Result += m_SharedWorker->DoWork (m_NestingLevel);
        }
    }
    void join (const RecursiveParReduceBodyWithSharedWorker & x) {
        m_Result += x.m_Result;
    }
    int result () { return m_Result; }
};

int Worker::DoWork ( int level ) {
    ++level;
    if ( level < max_nesting ) {
        RecursiveParReduceBodyWithSharedWorker rt (this, level);
        tbb::parallel_reduce (tbb::blocked_range<size_t>(0, reduce_range, reduce_grain), rt);
        return rt.result();
    }
    else
        return 1;
}

//! Regression test for hanging that occurred with the first version of cancellation propagation
void TestCancelation3 () {
    Worker w;
    int result   = w.DoWork (0);
    int expected = w.Validate(0);
    ASSERT ( result == expected, "Wrong calculation result");
}

struct StatsCounters {
    tbb::atomic<size_t> my_total_created;
    tbb::atomic<size_t> my_total_deleted;
    StatsCounters() {
        my_total_created = 0;
        my_total_deleted = 0;
    }
};

class ParReduceBody {
    StatsCounters* my_stats;
    size_t my_id;
    bool my_exception;

public:
    ParReduceBody( StatsCounters& s_, bool e_ ) : my_stats(&s_), my_exception(e_) {
        my_id = my_stats->my_total_created++;
    }

    ParReduceBody( const ParReduceBody& lhs ) {
        my_stats = lhs.my_stats;
        my_id = my_stats->my_total_created++;
    }

    ParReduceBody( ParReduceBody& lhs, tbb::split ) {
        my_stats = lhs.my_stats;
        my_id = my_stats->my_total_created++;
    }

    ~ParReduceBody(){ ++my_stats->my_total_deleted; }

    void operator()( const tbb::blocked_range<std::size_t>& /*range*/ ) const {
        //Do nothing, except for one task (chosen arbitrarily)
        if( my_id >= 12 ) {
            if( my_exception )
                ThrowTestException(1);
            else
                tbb::task::self().cancel_group_execution();
        }
    }

    void join( ParReduceBody& /*rhs*/ ) {}
};

void TestCancelation4() {
    StatsCounters statsObj;
    __TBB_TRY {
        tbb::task_group_context tgc1, tgc2;
        ParReduceBody body_for_cancellation(statsObj, false), body_for_exception(statsObj, true);
        tbb::parallel_reduce( tbb::blocked_range<std::size_t>(0,100000000,100), body_for_cancellation, tbb::simple_partitioner(), tgc1 );
        tbb::parallel_reduce( tbb::blocked_range<std::size_t>(0,100000000,100), body_for_exception, tbb::simple_partitioner(), tgc2 );
    } __TBB_CATCH(...) {}
    ASSERT ( statsObj.my_total_created==statsObj.my_total_deleted, "Not all parallel_reduce body objects created were reclaimed");
}

void RunParForAndReduceTests () {
    REMARK( "parallel for and reduce tests\n" );
    tbb::task_scheduler_init init (g_NumThreads);
    g_Master = Harness::CurrentTid();

#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    Test0();
    Test1();
    Test2();
    Test3();
    Test4();
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
    TestCancelation1();
    TestCancelation2();
    TestCancelation3();
    TestCancelation4();
}

////////////////////////////////////////////////////////////////////////////////
// Tests for tbb::parallel_do

#define ITER_RANGE          1000
#define ITEMS_TO_FEED       50
#define INNER_ITER_RANGE   100
#define OUTER_ITER_RANGE  50

#define PREPARE_RANGE(Iterator, rangeSize)  \
    size_t test_vector[rangeSize + 1]; \
    for (int i =0; i < rangeSize; i++) \
        test_vector[i] = i; \
    Iterator begin(&test_vector[0]); \
    Iterator end(&test_vector[rangeSize])

void Feed ( tbb::parallel_do_feeder<size_t> &feeder, size_t val ) {
    if (g_FedTasksCount < ITEMS_TO_FEED) {
        ++g_FedTasksCount;
        feeder.add(val);
    }
}

#include "harness_iterator.h"

#if TBB_USE_EXCEPTIONS

// Simple functor object with exception
class SimpleParDoBody {
public:
    void operator() ( size_t &value ) const {
        ++g_CurExecuted;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        Harness::ConcurrencyTracker ct;
        value += 1000;
        WaitUntilConcurrencyPeaks();
        ThrowTestException(1);
    }
};

// Simple functor object with exception and feeder
class SimpleParDoBodyWithFeeder : SimpleParDoBody {
public:
    void operator() ( size_t &value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        Feed(feeder, 0);
        SimpleParDoBody::operator()(value);
    }
};

// Tests exceptions without nesting
template <class Iterator, class simple_body>
void Test1_parallel_do () {
    ResetGlobals();
    PREPARE_RANGE(Iterator, ITER_RANGE);
    TRY();
        tbb::parallel_do<Iterator, simple_body>(begin, end, simple_body() );
    CATCH_AND_ASSERT();
    ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived cancellation");
    ASSERT (g_NumExceptionsCaught == 1, "No try_blocks in any body expected in this test");
    if ( !g_SolitaryException )
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");

} // void Test1_parallel_do ()

template <class Iterator>
class OuterParDoBody {
public:
    void operator()( size_t& /*value*/ ) const {
        ++g_OuterParCalls;
        PREPARE_RANGE(Iterator, INNER_ITER_RANGE);
        tbb::parallel_do<Iterator, SimpleParDoBody>(begin, end, SimpleParDoBody());
    }
};

template <class Iterator>
class OuterParDoBodyWithFeeder : OuterParDoBody<Iterator> {
public:
    void operator()( size_t& value, tbb::parallel_do_feeder<size_t>& feeder ) const {
        Feed(feeder, 0);
        OuterParDoBody<Iterator>::operator()(value);
    }
};

//! Uses parallel_do body containing an inner parallel_do with the default context not wrapped by a try-block.
/** Inner algorithms are spawned inside the new bound context by default. Since
    exceptions thrown from the inner parallel_do are not handled by the caller
    (outer parallel_do body) in this test, they will cancel all the sibling inner
    algorithms. **/
template <class Iterator, class outer_body>
void Test2_parallel_do () {
    ResetGlobals();
    PREPARE_RANGE(Iterator, ITER_RANGE);
    TRY();
        tbb::parallel_do<Iterator, outer_body >(begin, end, outer_body() );
    CATCH_AND_ASSERT();
    //if ( g_SolitaryException )
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived cancellation");
    ASSERT (g_NumExceptionsCaught == 1, "No try_blocks in any body expected in this test");
    if ( !g_SolitaryException )
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
} // void Test2_parallel_do ()

template <class Iterator>
class OuterParDoBodyWithIsolatedCtx {
public:
    void operator()( size_t& /*value*/ ) const {
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        ++g_OuterParCalls;
        PREPARE_RANGE(Iterator, INNER_ITER_RANGE);
        tbb::parallel_do<Iterator, SimpleParDoBody>(begin, end, SimpleParDoBody(), ctx);
    }
};

template <class Iterator>
class OuterParDoBodyWithIsolatedCtxWithFeeder : OuterParDoBodyWithIsolatedCtx<Iterator> {
public:
    void operator()( size_t& value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        Feed(feeder, 0);
        OuterParDoBodyWithIsolatedCtx<Iterator>::operator()(value);
    }
};

//! Uses parallel_do body invoking an inner parallel_do with an isolated context without a try-block.
/** Even though exceptions thrown from the inner parallel_do are not handled
    by the caller in this test, they will not affect sibling inner algorithms
    already running because of the isolated contexts. However because the first
    exception cancels the root parallel_do, at most the first g_NumThreads subranges
    will be processed (which launch inner parallel_dos) **/
template <class Iterator, class outer_body>
void Test3_parallel_do () {
    ResetGlobals();
    PREPARE_RANGE(Iterator, OUTER_ITER_RANGE);
    intptr_t innerCalls = INNER_ITER_RANGE,
             // The assumption here is the same as in outer parallel fors.
             minExecuted = (g_NumThreads - 1) * innerCalls;
    g_Master = Harness::CurrentTid();
    TRY();
        tbb::parallel_do<Iterator, outer_body >(begin, end, outer_body());
    CATCH_AND_ASSERT();
    // figure actual number of expected executions given the number of outer PDos started.
    minExecuted = (g_OuterParCalls - 1) * innerCalls;
    // one extra thread may run a task that sees cancellation.  Infrequent but possible
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");
    if(g_TGCCancelled > g_NumThreads) REMARK("Extra thread(s) executed after cancel (%d vs. %d)\n",
            (int)g_TGCCancelled, (int)g_NumThreads);
    if ( g_SolitaryException ) {
        ASSERT (g_CurExecuted > minExecuted, "Too few tasks survived exception");
        ASSERT (g_CurExecuted <= minExecuted + (g_ExecutedAtLastCatch + g_NumThreads), "Too many tasks survived exception");
    }
    ASSERT (g_NumExceptionsCaught == 1, "No try_blocks in any body expected in this test");
    if ( !g_SolitaryException )
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
} // void Test3_parallel_do ()

template <class Iterator>
class OuterParDoWithEhBody {
public:
    void operator()( size_t& /*value*/ ) const {
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        ++g_OuterParCalls;
        PREPARE_RANGE(Iterator, INNER_ITER_RANGE);
        TRY();
            tbb::parallel_do<Iterator, SimpleParDoBody>(begin, end, SimpleParDoBody(), ctx);
        CATCH();
    }
};

template <class Iterator>
class OuterParDoWithEhBodyWithFeeder : NoAssign, OuterParDoWithEhBody<Iterator> {
public:
    void operator()( size_t &value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        Feed(feeder, 0);
        OuterParDoWithEhBody<Iterator>::operator()(value);
    }
};

//! Uses parallel_for body invoking an inner parallel_for (with default bound context) inside a try-block.
/** Since exception(s) thrown from the inner parallel_for are handled by the caller
    in this test, they do not affect neither other tasks of the the root parallel_for
    nor sibling inner algorithms. **/
template <class Iterator, class outer_body_with_eh>
void Test4_parallel_do () {
    ResetGlobals( true, true );
    PREPARE_RANGE(Iterator, OUTER_ITER_RANGE);
    g_Master = Harness::CurrentTid();
    TRY();
        tbb::parallel_do<Iterator, outer_body_with_eh>(begin, end, outer_body_with_eh());
    CATCH();
    ASSERT (!l_ExceptionCaughtAtCurrentLevel, "All exceptions must have been handled in the parallel_do body");
    intptr_t innerCalls = INNER_ITER_RANGE,
             outerCalls = OUTER_ITER_RANGE + g_FedTasksCount,
             maxExecuted = outerCalls * innerCalls,
             minExecuted = 0;
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");
    if ( g_SolitaryException ) {
        minExecuted = maxExecuted - innerCalls;
        ASSERT (g_NumExceptionsCaught == 1, "No exception registered");
        ASSERT (g_CurExecuted >= minExecuted, "Too few tasks executed");
        // This test has the same property as Test4 (parallel_for); the exception can be
        // thrown, but some number of tasks from the outer Pdo can execute after the throw but
        // before the cancellation is signaled (have seen 36).
        ASSERT_WARNING(g_CurExecuted < maxExecuted || g_TGCCancelled, "All tasks survived exception. Oversubscription?");
    }
    else {
        minExecuted = g_NumExceptionsCaught;
        ASSERT (g_NumExceptionsCaught > 1 && g_NumExceptionsCaught <= outerCalls, "Unexpected actual number of exceptions");
        ASSERT (g_CurExecuted >= minExecuted, "Too many executed tasks reported");
        ASSERT (g_CurExecuted < g_ExecutedAtLastCatch + g_NumThreads + outerCalls, "Too many tasks survived multiple exceptions");
        ASSERT (g_CurExecuted <= outerCalls * (1 + g_NumThreads), "Too many tasks survived exception");
    }
} // void Test4_parallel_do ()

// This body throws an exception only if the task was added by feeder
class ParDoBodyWithThrowingFeederTasks {
public:
    //! This form of the function call operator can be used when the body needs to add more work during the processing
    void operator() ( size_t &value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        ++g_CurExecuted;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        Feed(feeder, 1);
        if (value == 1)
            ThrowTestException(1);
    }
}; // class ParDoBodyWithThrowingFeederTasks

// Test exception in task, which was added by feeder.
template <class Iterator>
void Test5_parallel_do () {
    ResetGlobals();
    PREPARE_RANGE(Iterator, ITER_RANGE);
    g_Master = Harness::CurrentTid();
    TRY();
        tbb::parallel_do<Iterator, ParDoBodyWithThrowingFeederTasks>(begin, end, ParDoBodyWithThrowingFeederTasks());
    CATCH();
    if (g_SolitaryException) {
        // Failure occurs when g_ExceptionInMaster is false, but all the 1 values in the range
        // are handled by the master thread.  In this case no throw occurs.
        ASSERT (l_ExceptionCaughtAtCurrentLevel     // we saw an exception
                || (!g_ExceptionInMaster && !g_NonMasterExecutedThrow) // non-master throws but none tried
                || (g_ExceptionInMaster && !g_MasterExecutedThrow)     // master throws but master didn't try
                , "At least one exception should occur");
        if(!g_ExceptionCaught) {
            if(g_ExceptionInMaster)
                REMARK("PDo exception not thrown; non-masters handled all throwing values.\n");
            else
                REMARK("PDo exception not thrown; master handled all throwing values.\n");
        }
    }
} // void Test5_parallel_do ()

#endif /* TBB_USE_EXCEPTIONS */

class ParDoBodyToCancel {
public:
    void operator()( size_t& /*value*/ ) const {
        ++g_CurExecuted;
        CancellatorTask::WaitUntilReady();
    }
};

class ParDoBodyToCancelWithFeeder : ParDoBodyToCancel {
public:
    void operator()( size_t& value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        Feed(feeder, 0);
        ParDoBodyToCancel::operator()(value);
    }
};

template<class B, class Iterator>
class ParDoWorkerTask : public tbb::task {
    tbb::task_group_context &my_ctx;

    tbb::task* execute () __TBB_override {
        PREPARE_RANGE(Iterator, INNER_ITER_RANGE);
        tbb::parallel_do<Iterator, B>( begin, end, B(), my_ctx );
        return NULL;
    }
public:
    ParDoWorkerTask ( tbb::task_group_context& ctx ) : my_ctx(ctx) {}
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
template <class Iterator, class body_to_cancel>
void TestCancelation1_parallel_do () {
    ResetGlobals( false );
    intptr_t  threshold = 10;
    tbb::task_group_context  ctx;
    ctx.reset();
    tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r.set_ref_count(3);
    r.spawn( *new( r.allocate_child() ) CancellatorTask(ctx, threshold) );
    __TBB_Yield();
    r.spawn( *new( r.allocate_child() ) ParDoWorkerTask<body_to_cancel, Iterator>(ctx) );
    TRY();
        r.wait_for_all();
    CATCH_AND_FAIL();
    ASSERT (g_CurExecuted < g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks were executed after cancellation");
    r.destroy(r);
}

class ParDoBodyToCancel2 {
public:
    void operator()( size_t& /*value*/ ) const {
        ++g_CurExecuted;
        Harness::ConcurrencyTracker ct;
        // The test will hang (and be timed out by the test system) if is_cancelled() is broken
        while( !tbb::task::self().is_cancelled() )
            __TBB_Yield();
    }
};

class ParDoBodyToCancel2WithFeeder : ParDoBodyToCancel2 {
public:
    void operator()( size_t& value, tbb::parallel_do_feeder<size_t> &feeder ) const {
        Feed(feeder, 0);
        ParDoBodyToCancel2::operator()(value);
    }
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
/** This version also tests task::is_cancelled() method. **/
template <class Iterator, class body_to_cancel>
void TestCancelation2_parallel_do () {
    ResetGlobals();
    RunCancellationTest<ParDoWorkerTask<body_to_cancel, Iterator>, CancellatorTask2>();
}

#define RunWithSimpleBody(func, body)       \
    func<Harness::RandomIterator<size_t>, body>();           \
    func<Harness::RandomIterator<size_t>, body##WithFeeder>();  \
    func<Harness::ForwardIterator<size_t>, body>();         \
    func<Harness::ForwardIterator<size_t>, body##WithFeeder>()

#define RunWithTemplatedBody(func, body)       \
    func<Harness::RandomIterator<size_t>, body<Harness::RandomIterator<size_t> > >();           \
    func<Harness::RandomIterator<size_t>, body##WithFeeder<Harness::RandomIterator<size_t> > >();  \
    func<Harness::ForwardIterator<size_t>, body<Harness::ForwardIterator<size_t> > >();         \
    func<Harness::ForwardIterator<size_t>, body##WithFeeder<Harness::ForwardIterator<size_t> > >()

void RunParDoTests() {
    REMARK( "parallel do tests\n" );
    tbb::task_scheduler_init init (g_NumThreads);
    g_Master = Harness::CurrentTid();
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    RunWithSimpleBody(Test1_parallel_do, SimpleParDoBody);
    RunWithTemplatedBody(Test2_parallel_do, OuterParDoBody);
    RunWithTemplatedBody(Test3_parallel_do, OuterParDoBodyWithIsolatedCtx);
    RunWithTemplatedBody(Test4_parallel_do, OuterParDoWithEhBody);
    Test5_parallel_do<Harness::ForwardIterator<size_t> >();
    Test5_parallel_do<Harness::RandomIterator<size_t> >();
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
    RunWithSimpleBody(TestCancelation1_parallel_do, ParDoBodyToCancel);
    RunWithSimpleBody(TestCancelation2_parallel_do, ParDoBodyToCancel2);
}

////////////////////////////////////////////////////////////////////////////////
// Tests for tbb::pipeline

#define NUM_ITEMS   100

const size_t c_DataEndTag = size_t(~0);

int g_NumTokens = 0;

// Simple input filter class, it assigns 1 to all array members
// It stops when it receives item equal to -1
class InputFilter: public tbb::filter {
    tbb::atomic<size_t> m_Item;
    size_t m_Buffer[NUM_ITEMS + 1];
public:
    InputFilter() : tbb::filter(parallel) {
        m_Item = 0;
        for (size_t i = 0; i < NUM_ITEMS; ++i )
            m_Buffer[i] = 1;
        m_Buffer[NUM_ITEMS] = c_DataEndTag;
    }

    void* operator()( void* ) __TBB_override {
        size_t item = m_Item.fetch_and_increment();
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        if(item == 1) {
            ++g_PipelinesStarted;   // count on emitting the first item.
        }
        if ( item >= NUM_ITEMS )
            return NULL;
        m_Buffer[item] = 1;
        return &m_Buffer[item];
    }

    size_t* buffer() { return m_Buffer; }
}; // class InputFilter

// Pipeline filter, without exceptions throwing
class NoThrowFilter : public tbb::filter {
    size_t m_Value;
public:
    enum operation {
        addition,
        subtraction,
        multiplication
    } m_Operation;

    NoThrowFilter(operation _operation, size_t value, bool is_parallel)
        : filter(is_parallel? tbb::filter::parallel : tbb::filter::serial_in_order),
        m_Value(value), m_Operation(_operation)
    {}
    void* operator()(void* item) __TBB_override {
        size_t &value = *(size_t*)item;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        ASSERT(value != c_DataEndTag, "terminator element is being processed");
        switch (m_Operation){
            case addition:
                value += m_Value;
                break;
            case subtraction:
                value -= m_Value;
                break;
            case multiplication:
                value *= m_Value;
                break;
            default:
                ASSERT(0, "Wrong operation parameter passed to NoThrowFilter");
        } // switch (m_Operation)
        return item;
    }
};

// Test pipeline without exceptions throwing
void Test0_pipeline () {
    ResetGlobals();
    // Run test when serial filter is the first non-input filter
    InputFilter inputFilter;  //Emits NUM_ITEMS items
    NoThrowFilter filter1(NoThrowFilter::addition, 99, false);
    NoThrowFilter filter2(NoThrowFilter::subtraction, 90, true);
    NoThrowFilter filter3(NoThrowFilter::multiplication, 5, false);
    // Result should be 50 for all items except the last
    tbb::pipeline p;
    p.add_filter(inputFilter);
    p.add_filter(filter1);
    p.add_filter(filter2);
    p.add_filter(filter3);
    p.run(8);
    for (size_t i = 0; i < NUM_ITEMS; ++i)
        ASSERT(inputFilter.buffer()[i] == 50, "pipeline didn't process items properly");
} // void Test0_pipeline ()

#if TBB_USE_EXCEPTIONS

// Simple filter with exception throwing.  If parallel, will wait until
// as many parallel filters start as there are threads.
class SimpleFilter : public tbb::filter {
    bool m_canThrow;
public:
    SimpleFilter (tbb::filter::mode _mode, bool canThrow ) : filter (_mode), m_canThrow(canThrow) {}
    void* operator()(void* item) __TBB_override {
        ++g_CurExecuted;
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled() ) ++g_TGCCancelled;
        if ( m_canThrow ) {
            if ( !is_serial() ) {
                Harness::ConcurrencyTracker ct;
                WaitUntilConcurrencyPeaks( min(g_NumTokens, g_NumThreads) );
            }
            ThrowTestException(1);
        }
        return item;
    }
}; // class SimpleFilter

// This enumeration represents filters order in pipeline
struct FilterSet {
    tbb::filter::mode   mode1,
                        mode2;
    bool                throw1,
                        throw2;

    FilterSet( tbb::filter::mode m1, tbb::filter::mode m2, bool t1, bool t2 )
        : mode1(m1), mode2(m2), throw1(t1), throw2(t2)
    {}
}; // struct FilterSet

FilterSet serial_parallel( tbb::filter::serial, tbb::filter::parallel, /*throw1*/false, /*throw2*/true );

template<typename InFilter, typename Filter>
class CustomPipeline : protected tbb::pipeline {
    InFilter inputFilter;
    Filter filter1;
    Filter filter2;
public:
    CustomPipeline( const FilterSet& filters )
        : filter1(filters.mode1, filters.throw1), filter2(filters.mode2, filters.throw2)
    {
       add_filter(inputFilter);
       add_filter(filter1);
       add_filter(filter2);
    }
    void run () { tbb::pipeline::run(g_NumTokens); }
    void run ( tbb::task_group_context& ctx ) { tbb::pipeline::run(g_NumTokens, ctx); }

    using tbb::pipeline::add_filter;
};

typedef CustomPipeline<InputFilter, SimpleFilter> SimplePipeline;

// Tests exceptions without nesting
void Test1_pipeline ( const FilterSet& filters ) {
    ResetGlobals();
    SimplePipeline testPipeline(filters);
    TRY();
        testPipeline.run();
        if ( g_CurExecuted == 2 * NUM_ITEMS ) {
            // all the items were processed, though an exception was supposed to occur.
            if(!g_ExceptionInMaster && g_NonMasterExecutedThrow > 0) {
                // if !g_ExceptionInMaster, the master thread is not allowed to throw.
                // if g_nonMasterExcutedThrow > 0 then a thread besides the master tried to throw.
                ASSERT(filters.mode1 != tbb::filter::parallel && filters.mode2 != tbb::filter::parallel, "Unusual count");
            }
            else {
                REMARK("test1_Pipeline with %d threads: Only the master thread tried to throw, and it is not allowed to.\n", (int)g_NumThreads);
            }
            // In case of all serial filters they might be all executed in the thread(s)
            // where exceptions are not allowed by the common test logic. So we just quit.
            return;
        }
    CATCH_AND_ASSERT();
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");
    ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
    ASSERT (g_NumExceptionsCaught == 1, "No try_blocks in any body expected in this test");
    if ( !g_SolitaryException )
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");

} // void Test1_pipeline ()

// Filter with nesting
class OuterFilter : public tbb::filter {
public:
    OuterFilter (tbb::filter::mode _mode, bool ) : filter (_mode) {}

    void* operator()(void* item) __TBB_override {
        ++g_OuterParCalls;
        SimplePipeline testPipeline(serial_parallel);
        testPipeline.run();
        return item;
    }
}; // class OuterFilter

//! Uses pipeline containing an inner pipeline with the default context not wrapped by a try-block.
/** Inner algorithms are spawned inside the new bound context by default. Since
    exceptions thrown from the inner pipeline are not handled by the caller
    (outer pipeline body) in this test, they will cancel all the sibling inner
    algorithms. **/
void Test2_pipeline ( const FilterSet& filters ) {
    ResetGlobals();
    g_NestedPipelines = true;
    CustomPipeline<InputFilter, OuterFilter> testPipeline(filters);
    TRY();
        testPipeline.run();
    CATCH_AND_ASSERT();
    bool okayNoExceptionCaught = (g_ExceptionInMaster && !g_MasterExecutedThrow) || (!g_ExceptionInMaster && !g_NonMasterExecutedThrow);
    ASSERT (g_NumExceptionsCaught == 1 || okayNoExceptionCaught, "No try_blocks in any body expected in this test");
    if ( g_SolitaryException ) {
        if( g_TGCCancelled > g_NumThreads) REMARK( "Extra tasks ran after exception thrown (%d vs. %d)\n",
                (int)g_TGCCancelled, (int)g_NumThreads);
    }
    else {
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived exception");
    }
} // void Test2_pipeline ()

//! creates isolated inner pipeline and runs it.
class OuterFilterWithIsolatedCtx : public tbb::filter {
public:
    OuterFilterWithIsolatedCtx(tbb::filter::mode m, bool ) : filter(m) {}

    void* operator()(void* item) __TBB_override {
        ++g_OuterParCalls;
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        // create inner pipeline with serial input, parallel output filter, second filter throws
        SimplePipeline testPipeline(serial_parallel);
        testPipeline.run(ctx);
        return item;
    }
}; // class OuterFilterWithIsolatedCtx

//! Uses pipeline invoking an inner pipeline with an isolated context without a try-block.
/** Even though exceptions thrown from the inner pipeline are not handled
    by the caller in this test, they will not affect sibling inner algorithms
    already running because of the isolated contexts. However because the first
    exception cancels the root parallel_do only the first g_NumThreads subranges
    will be processed (which launch inner pipelines) **/
void Test3_pipeline ( const FilterSet& filters ) {
    for( int nTries = 1; nTries <= 4; ++nTries) {
        ResetGlobals();
        g_NestedPipelines = true;
        g_Master = Harness::CurrentTid();
        intptr_t innerCalls = NUM_ITEMS,
                 minExecuted = (g_NumThreads - 1) * innerCalls;
        CustomPipeline<InputFilter, OuterFilterWithIsolatedCtx> testPipeline(filters);
        TRY();
            testPipeline.run();
        CATCH_AND_ASSERT();

        bool okayNoExceptionCaught = (g_ExceptionInMaster && !g_MasterExecuted) ||
            (!g_ExceptionInMaster && !g_NonMasterExecuted);
        // only test assertions if the test threw an exception (or we don't care)
        bool testSucceeded = okayNoExceptionCaught || g_NumExceptionsCaught > 0;
        if(testSucceeded) {
            if (g_SolitaryException) {

                // The test is one outer pipeline with two NestedFilters that each start an inner pipeline.
                // Each time the input filter of a pipeline delivers its first item, it increments
                // g_PipelinesStarted.  When g_SolitaryException, the throw will not occur until
                // g_PipelinesStarted >= 3.  (This is so at least a second pipeline in its own isolated
                // context will start; that is what we're testing.)
                //
                // There are two pipelines which will NOT run to completion when a solitary throw
                // happens in an isolated inner context: the outer pipeline and the pipeline which
                // throws.  All the other pipelines which start should run to completion.  But only
                // inner body invocations are counted.
                //
                // So g_CurExecuted should be about
                //
                //   (2*NUM_ITEMS) * (g_PipelinesStarted - 2) + 1
                //   ^ executions for each completed pipeline
                //                   ^ completing pipelines (remembering two will not complete)
                //                                              ^ one for the inner throwing pipeline

                minExecuted = (2*NUM_ITEMS) * (g_PipelinesStarted - 2) + 1;
                // each failing pipeline must execute at least two tasks
                ASSERT(g_CurExecuted >= minExecuted, "Too few tasks survived exception");
                // no more than g_NumThreads tasks will be executed in a cancelled context.  Otherwise
                // tasks not executing at throw were scheduled.
                ASSERT( g_TGCCancelled <= g_NumThreads, "Tasks not in-flight were executed");
                ASSERT(g_NumExceptionsCaught == 1, "Should have only one exception");
                // if we're only throwing from the master thread, and that thread didn't
                // participate in the pipelines, then no throw occurred.
                if(g_ExceptionInMaster && !g_MasterExecuted) {
                    REMARK_ONCE("Master expected to throw, but didn't participate.\n");
                }
                else if(!g_ExceptionInMaster && !g_NonMasterExecuted) {
                    REMARK_ONCE("Non-master expected to throw, but didn't participate.\n");
                }
            }
            ASSERT (g_NumExceptionsCaught == 1 || okayNoExceptionCaught, "No try_blocks in any body expected in this test");
            ASSERT ((g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads) || okayNoExceptionCaught, "Too many tasks survived exception");
            if(nTries > 1) REMARK("Test3_pipeline succeeeded on try %d\n", nTries);
            return;
        }
    }
    REMARK_ONCE("Test3_pipeline failed for g_NumThreads==%d, g_ExceptionInMaster==%s , g_SolitaryException==%s\n",
            g_NumThreads, g_ExceptionInMaster?"T":"F", g_SolitaryException?"T":"F");
} // void Test3_pipeline ()

class OuterFilterWithEhBody : public tbb::filter {
public:
    OuterFilterWithEhBody(tbb::filter::mode m, bool ) : filter(m) {}

    void* operator()(void* item) __TBB_override {
        tbb::task_group_context ctx(tbb::task_group_context::isolated);
        ++g_OuterParCalls;
        SimplePipeline testPipeline(serial_parallel);
        TRY();
            testPipeline.run(ctx);
        CATCH();
        return item;
    }
}; // class OuterFilterWithEhBody

//! Uses pipeline body invoking an inner pipeline (with isolated context) inside a try-block.
/** Since exception(s) thrown from the inner pipeline are handled by the caller
    in this test, they do not affect other tasks of the the root pipeline
    nor sibling inner algorithms. **/
void Test4_pipeline ( const FilterSet& filters ) {
#if __GNUC__ && !__INTEL_COMPILER
    if ( strncmp(__VERSION__, "4.1.0", 5) == 0 ) {
        REMARK_ONCE("Known issue: one of exception handling tests is skipped.\n");
        return;
    }
#endif
    ResetGlobals( true, true );
    // each outer pipeline stage will start NUM_ITEMS inner pipelines.
    // each inner pipeline that doesn't throw will process NUM_ITEMS items.
    // for solitary exception there will be one pipeline that only processes one stage, one item.
    // innerCalls should be 2*NUM_ITEMS
    intptr_t innerCalls = 2*NUM_ITEMS,
             outerCalls = 2 * NUM_ITEMS,
             maxExecuted = outerCalls * innerCalls;  // the number of invocations of the inner pipelines
    CustomPipeline<InputFilter, OuterFilterWithEhBody> testPipeline(filters);
    TRY();
        testPipeline.run();
    CATCH_AND_ASSERT();
    intptr_t  minExecuted = 0;
    bool okayNoExceptionCaught = (g_ExceptionInMaster && !g_MasterExecuted) ||
        (!g_ExceptionInMaster && !g_NonMasterExecuted);
    if ( g_SolitaryException ) {
        minExecuted = maxExecuted - innerCalls;  // one throwing inner pipeline
        ASSERT (g_NumExceptionsCaught == 1 || okayNoExceptionCaught, "No exception registered");
        ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived exception");  // probably will assert.
    }
    else {
        // we assume throwing pipelines will not count
        minExecuted = (outerCalls - g_NumExceptionsCaught) * innerCalls;
        ASSERT((g_NumExceptionsCaught >= 1 && g_NumExceptionsCaught <= outerCalls)||okayNoExceptionCaught, "Unexpected actual number of exceptions");
        ASSERT (g_CurExecuted >= minExecuted, "Too many executed tasks reported");
        // too many already-scheduled tasks are started after the first exception is
        // thrown.  And g_ExecutedAtLastCatch is updated every time an exception is caught.
        // So with multiple exceptions there are a variable number of tasks that have been
        // discarded because of the signals.
        // each throw is caught, so we will see many cancelled tasks.  g_ExecutedAtLastCatch is
        // updated with each throw, so the value will be the number of tasks executed at the last
        ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks survived multiple exceptions");
    }
} // void Test4_pipeline ()

//! Testing filter::finalize method
#define BUFFER_SIZE     32
#define NUM_BUFFERS     1024

tbb::atomic<size_t> g_AllocatedCount; // Number of currently allocated buffers
tbb::atomic<size_t> g_TotalCount; // Total number of allocated buffers

//! Base class for all filters involved in finalize method testing
class FinalizationBaseFilter : public tbb::filter {
public:
    FinalizationBaseFilter ( tbb::filter::mode m ) : filter(m) {}

    // Deletes buffers if exception occurred
    virtual void finalize( void* item ) __TBB_override {
        size_t* m_Item = (size_t*)item;
        delete[] m_Item;
        --g_AllocatedCount;
    }
};

//! Input filter to test finalize method
class InputFilterWithFinalization: public FinalizationBaseFilter {
public:
    InputFilterWithFinalization() : FinalizationBaseFilter(tbb::filter::serial) {
        g_TotalCount = 0;
    }
    void* operator()( void* ) __TBB_override {
        if (g_TotalCount == NUM_BUFFERS)
            return NULL;
        size_t* item = new size_t[BUFFER_SIZE];
        for (int i = 0; i < BUFFER_SIZE; i++)
            item[i] = 1;
        ++g_TotalCount;
        ++g_AllocatedCount;
        return item;
    }
};

// The filter multiplies each buffer item by 10.
class ProcessingFilterWithFinalization : public FinalizationBaseFilter {
public:
    ProcessingFilterWithFinalization (tbb::filter::mode _mode, bool) : FinalizationBaseFilter (_mode) {}

    void* operator()( void* item) __TBB_override {
        if(g_Master == Harness::CurrentTid()) g_MasterExecuted = true;
        else g_NonMasterExecuted = true;
        if( tbb::task::self().is_cancelled()) ++g_TGCCancelled;
        if (g_TotalCount > NUM_BUFFERS / 2)
            ThrowTestException(1);
        size_t* m_Item = (size_t*)item;
        for (int i = 0; i < BUFFER_SIZE; i++)
            m_Item[i] *= 10;
        return item;
    }
};

// Output filter deletes previously allocated buffer
class OutputFilterWithFinalization : public FinalizationBaseFilter {
public:
    OutputFilterWithFinalization (tbb::filter::mode m) : FinalizationBaseFilter (m) {}

    void* operator()( void* item) __TBB_override {
        size_t* m_Item = (size_t*)item;
        delete[] m_Item;
        --g_AllocatedCount;
        return NULL;
    }
};

//! Tests filter::finalize method
void Test5_pipeline ( const FilterSet& filters ) {
    ResetGlobals();
    g_AllocatedCount = 0;
    CustomPipeline<InputFilterWithFinalization, ProcessingFilterWithFinalization> testPipeline(filters);
    OutputFilterWithFinalization my_output_filter(tbb::filter::parallel);

    testPipeline.add_filter(my_output_filter);
    TRY();
        testPipeline.run();
    CATCH();
    ASSERT (g_AllocatedCount == 0, "Memory leak: Some my_object weren't destroyed");
} // void Test5_pipeline ()

//! Tests pipeline function passed with different combination of filters
template<void testFunc(const FilterSet&)>
void TestWithDifferentFilters() {
    const int NumFilterTypes = 3;
    const tbb::filter::mode modes[NumFilterTypes] = {
            tbb::filter::parallel,
            tbb::filter::serial,
            tbb::filter::serial_out_of_order
        };
    for ( int i = 0; i < NumFilterTypes; ++i ) {
        for ( int j = 0; j < NumFilterTypes; ++j ) {
            for ( int k = 0; k < 2; ++k )
                testFunc( FilterSet(modes[i], modes[j], k == 0, k != 0) );
        }
    }
}

#endif /* TBB_USE_EXCEPTIONS */

class FilterToCancel : public tbb::filter {
public:
    FilterToCancel(bool is_parallel)
        : filter( is_parallel ? tbb::filter::parallel : tbb::filter::serial_in_order )
    {}
    void* operator()(void* item) __TBB_override {
        ++g_CurExecuted;
        CancellatorTask::WaitUntilReady();
        return item;
    }
}; // class FilterToCancel

template <class Filter_to_cancel>
class PipelineLauncherTask : public tbb::task {
    tbb::task_group_context &my_ctx;
public:
    PipelineLauncherTask ( tbb::task_group_context& ctx ) : my_ctx(ctx) {}

    tbb::task* execute () __TBB_override {
        // Run test when serial filter is the first non-input filter
        InputFilter inputFilter;
        Filter_to_cancel filterToCancel(true);
        tbb::pipeline p;
        p.add_filter(inputFilter);
        p.add_filter(filterToCancel);
        p.run(g_NumTokens, my_ctx);
        return NULL;
    }
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
void TestCancelation1_pipeline () {
    ResetGlobals();
    g_ThrowException = false;
    intptr_t  threshold = 10;
    tbb::task_group_context ctx;
    ctx.reset();
    tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r.set_ref_count(3);
    r.spawn( *new( r.allocate_child() ) CancellatorTask(ctx, threshold) );
    __TBB_Yield();
    r.spawn( *new( r.allocate_child() ) PipelineLauncherTask<FilterToCancel>(ctx) );
    TRY();
        r.wait_for_all();
    CATCH_AND_FAIL();
    r.destroy(r);
    ASSERT( g_TGCCancelled <= g_NumThreads, "Too many tasks survived cancellation");
    ASSERT (g_CurExecuted < g_ExecutedAtLastCatch + g_NumThreads, "Too many tasks were executed after cancellation");
}

class FilterToCancel2 : public tbb::filter {
public:
    FilterToCancel2(bool is_parallel)
        : filter ( is_parallel ? tbb::filter::parallel : tbb::filter::serial_in_order)
    {}

    void* operator()(void* item) __TBB_override {
        ++g_CurExecuted;
        Harness::ConcurrencyTracker ct;
        // The test will hang (and be timed out by the test system) if is_cancelled() is broken
        while( !tbb::task::self().is_cancelled() )
            __TBB_Yield();
        return item;
    }
};

//! Test for cancelling an algorithm from outside (from a task running in parallel with the algorithm).
/** This version also tests task::is_cancelled() method. **/
void TestCancelation2_pipeline () {
    ResetGlobals();
    RunCancellationTest<PipelineLauncherTask<FilterToCancel2>, CancellatorTask2>();
    // g_CurExecuted is always >= g_ExecutedAtLastCatch, because the latter is always a snapshot of the
    // former, and g_CurExecuted is monotonic increasing.  so the comparison should be at least ==.
    // If another filter is started after cancel but before cancellation is propagated, then the
    // number will be larger.
    ASSERT (g_CurExecuted <= g_ExecutedAtLastCatch, "Some tasks were executed after cancellation");
}

void RunPipelineTests() {
    REMARK( "pipeline tests\n" );
    tbb::task_scheduler_init init (g_NumThreads);
    g_Master = Harness::CurrentTid();
    g_NumTokens = 2 * g_NumThreads;

    Test0_pipeline();
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    TestWithDifferentFilters<Test1_pipeline>();
    TestWithDifferentFilters<Test2_pipeline>();
    TestWithDifferentFilters<Test3_pipeline>();
    TestWithDifferentFilters<Test4_pipeline>();
    TestWithDifferentFilters<Test5_pipeline>();
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
    TestCancelation1_pipeline();
    TestCancelation2_pipeline();
}


#if TBB_USE_EXCEPTIONS

class MyCapturedException : public tbb::captured_exception {
public:
    static int m_refCount;

    MyCapturedException () : tbb::captured_exception("MyCapturedException", "test") { ++m_refCount; }
    ~MyCapturedException () throw() { --m_refCount; }

    MyCapturedException* move () throw() __TBB_override {
        MyCapturedException* movee = (MyCapturedException*)malloc(sizeof(MyCapturedException));
        return ::new (movee) MyCapturedException;
    }
    void destroy () throw() __TBB_override {
        this->~MyCapturedException();
        free(this);
    }
    void operator delete ( void* p ) { free(p); }
};

int MyCapturedException::m_refCount = 0;

void DeleteTbbException ( volatile tbb::tbb_exception* pe ) {
    delete pe;
}

void TestTbbExceptionAPI () {
    const char *name = "Test captured exception",
               *reason = "Unit testing";
    tbb::captured_exception e(name, reason);
    ASSERT (strcmp(e.name(), name) == 0, "Setting captured exception name failed");
    ASSERT (strcmp(e.what(), reason) == 0, "Setting captured exception reason failed");
    tbb::captured_exception c(e);
    ASSERT (strcmp(c.name(), e.name()) == 0, "Copying captured exception name failed");
    ASSERT (strcmp(c.what(), e.what()) == 0, "Copying captured exception reason failed");
    tbb::captured_exception *m = e.move();
    ASSERT (strcmp(m->name(), name) == 0, "Moving captured exception name failed");
    ASSERT (strcmp(m->what(), reason) == 0, "Moving captured exception reason failed");
    ASSERT (!e.name() && !e.what(), "Moving semantics broken");
    m->destroy();

    MyCapturedException mce;
    MyCapturedException *mmce = mce.move();
    ASSERT( MyCapturedException::m_refCount == 2, NULL );
    DeleteTbbException(mmce);
    ASSERT( MyCapturedException::m_refCount == 1, NULL );
}

#endif /* TBB_USE_EXCEPTIONS */

/** If min and max thread numbers specified on the command line are different,
    the test is run only for 2 sizes of the thread pool (MinThread and MaxThread)
    to be able to test the high and low contention modes while keeping the test reasonably fast **/
int TestMain () {
    if(tbb::task_scheduler_init::default_num_threads() == 1) {
        REPORT("Known issue: tests require multiple hardware threads\n");
        return Harness::Skipped;
    }
    REMARK ("Using %s\n", TBB_USE_CAPTURED_EXCEPTION ? "tbb:captured_exception" : "exact exception propagation");
    MinThread = min(tbb::task_scheduler_init::default_num_threads(), max(2, MinThread));
    MaxThread = max(MinThread, min(tbb::task_scheduler_init::default_num_threads(), MaxThread));
    ASSERT (FLAT_RANGE >= FLAT_GRAIN * MaxThread, "Fix defines");
    int step = max((MaxThread - MinThread + 1)/2, 1);
    for ( g_NumThreads = MinThread; g_NumThreads <= MaxThread; g_NumThreads += step ) {
        REMARK ("Number of threads %d\n", g_NumThreads);
        // Execute in all the possible modes
        for ( size_t j = 0; j < 4; ++j ) {
            g_ExceptionInMaster = (j & 1) != 0;
            g_SolitaryException = (j & 2) != 0;
            REMARK("g_ExceptionInMaster==%s, g_SolitaryException==%s\n", g_ExceptionInMaster?"T":"F", g_SolitaryException?"T":"F");
            RunParForAndReduceTests();
            RunParDoTests();
            RunPipelineTests();
        }
    }
#if TBB_USE_EXCEPTIONS
    TestTbbExceptionAPI();
#endif
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception handling tests are skipped.\n");
#endif
    return Harness::Done;
}

#else /* !__TBB_TASK_GROUP_CONTEXT */

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_TASK_GROUP_CONTEXT */
