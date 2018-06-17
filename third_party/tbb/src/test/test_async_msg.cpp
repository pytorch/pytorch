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

#ifndef TBB_PREVIEW_FLOW_GRAPH_FEATURES
    #define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#endif

#include "tbb/tbb_config.h"

#if __TBB_PREVIEW_ASYNC_MSG

#if _MSC_VER
#pragma warning (disable: 4503) // Suppress "decorated name length exceeded, name was truncated" warning
#endif

#include "tbb/flow_graph.h"
#include "tbb/tbb_thread.h"
#include "tbb/concurrent_queue.h"

#include "harness.h"
#include "harness_graph.h"
#include "harness_barrier.h"

#include <sstream>      // std::ostringstream
#include <type_traits>  // std::is_base_of

static const int USE_N = 1000;
static const int ACTIVITY_PAUSE_MS_NODE1 = 0;//500;
static const int ACTIVITY_PAUSE_MS_NODE2 = 0;//100;

#define _TRACE_(msg) {                                                  \
    if (Verbose) {                                                      \
        std::ostringstream os;                                          \
        os << "[TID=" << tbb::this_tbb_thread::get_id() << "] " << msg; \
        REMARK("%s\n", os.str().c_str());                               \
    }                                                                   \
}

class UserAsyncActivity // Singleton
{
public:
    static UserAsyncActivity* create(const tbb::flow::async_msg<int>& msg, int timeoutMS) {
        ASSERT(s_Activity == NULL, "created twice");
        _TRACE_( "Create UserAsyncActivity" );
        s_Activity = new UserAsyncActivity(msg, timeoutMS);
        _TRACE_( "CREATED! UserAsyncActivity" );
        return s_Activity;
    }

    static void destroy() {
        _TRACE_( "Start UserAsyncActivity::destroy()" );
        ASSERT(s_Activity != NULL, "destroyed twice");
        s_Activity->myThread.join();
        delete s_Activity;
        s_Activity = NULL;
        _TRACE_( "End UserAsyncActivity::destroy()" );
    }

    static int s_Result;

private:
    static void threadFunc(UserAsyncActivity* activity) {
        _TRACE_( "UserAsyncActivity::threadFunc" );

        Harness::Sleep(activity->myTimeoutMS);

        const int result = static_cast<int>(reinterpret_cast<size_t>(activity)) & 0xFF; // just different random results
        s_Result = result;

        _TRACE_( "UserAsyncActivity::threadFunc - returned result " << result );

        activity->returnActivityResults(result);
    }

    UserAsyncActivity(const tbb::flow::async_msg<int>& msg, int timeoutMS) : myMsg(msg), myTimeoutMS(timeoutMS)
        , myThread(threadFunc, this)
    {
        // Start local thread here...
        _TRACE_( "Started AsyncActivity" );
    }

    // Will be called from working thread
    void returnActivityResults(int result) {
        myMsg.set(result);
    }

private: // DATA
    tbb::flow::async_msg<int>   myMsg;
    int                         myTimeoutMS;
    tbb::tbb_thread             myThread;

    static UserAsyncActivity*   s_Activity;
};

UserAsyncActivity* UserAsyncActivity::s_Activity = NULL;
int UserAsyncActivity::s_Result = -1;

class UserAsyncMsg1 : public tbb::flow::async_msg<int>
{
public:
    typedef tbb::flow::async_msg<int> base;

    UserAsyncMsg1() : base() {}
    UserAsyncMsg1(int value) : base(value) {}
    UserAsyncMsg1(const UserAsyncMsg1& msg) : base(msg) {}
};

struct F2_body : tbb::internal::no_assign
{
    static int          s_FinalResult;

    int&                myI;
    bool                myAlive;

    F2_body(int& i) : myI(i), myAlive(true) {}

    F2_body(const F2_body& b) : myI(b.myI), myAlive(true) {}

    ~F2_body() {
        myAlive = false;
        _TRACE_( "~F2_body" );
    }

    void operator () (int result) {
        __TBB_ASSERT(myAlive, "dead node");

        // Handle async activity result here
        s_FinalResult = result;
        _TRACE_( "F2: Got async_msg result = " << result );
    }
};

// static
int F2_body::s_FinalResult = -2;

static bool testSimplestCase() {
    bool bOk = true;
    _TRACE_( "--- SAMPLE 1 (simple case 3-in-1: F1(A<T>) ---> F2(T)) " );

    for (int i = 0; i <= 2; ++i) {
        _TRACE_( "CASE " << i + 1 << ": data is " << (i > 0 ? "NOT " : "") << "ready in storage" << (i > 1 ? " NO WAITING in graph" : "") );
        _TRACE_( "MAIN THREAD" );

        {
            tbb::flow::graph g;
            tbb::flow::function_node< tbb::flow::continue_msg, UserAsyncMsg1 > f1( g, tbb::flow::unlimited,
                [&]( tbb::flow::continue_msg ) -> UserAsyncMsg1 {
                    _TRACE_( "F1: Created async_msg" );

                    UserAsyncMsg1 a;
                    UserAsyncActivity::create(a, (i == 0 ? 0 : 1)*ACTIVITY_PAUSE_MS_NODE1);

                    Harness::Sleep(ACTIVITY_PAUSE_MS_NODE2); // let activity to finish
                    return a;
                }
            );


            tbb::flow::function_node< int > f2( g, tbb::flow::unlimited,
                F2_body(i)
            );

            make_edge(f1, f2);
            f1.try_put( tbb::flow::continue_msg() );
            g.wait_for_all();
            UserAsyncActivity::destroy();
            _TRACE_( "Done UserAsyncActivity::destroy" );
            g.wait_for_all();
            _TRACE_( "Done g.wait_for_all()" );
        }

        _TRACE_( "--- THE END --- " );

        if (F2_body::s_FinalResult >= 0 && UserAsyncActivity::s_Result == F2_body::s_FinalResult) {
            _TRACE_( "CASE " << i + 1 << ": " << "PASSED" );
        }
        else {
            _TRACE_( "CASE " << i + 1 << ": " << "FAILED! " << UserAsyncActivity::s_Result << " != " << F2_body::s_FinalResult );
            bOk = false;
            ASSERT(0, "testSimplestCase failed");
        }
    }

    return bOk;
}

// ========================================================

class UserAsyncActivityChaining;

class UserAsyncMsg : public tbb::flow::async_msg<int>
{
public:
    typedef tbb::flow::async_msg<int> base;

    UserAsyncMsg() : base() {}
    UserAsyncMsg(int value) : base(value) {}
    UserAsyncMsg(const UserAsyncMsg& msg) : base(msg) {}

    // Notify AsyncActivity that it must return result because async calculation chain is over
    void finalize() const __TBB_override;
};

class UserAsyncActivityChaining // Singleton: task queue in worker thread
{
public:
    static UserAsyncActivityChaining* instance() {
        if (s_Activity == NULL) {
            s_Activity = new UserAsyncActivityChaining();
        }

        return s_Activity;
    }

    static void destroy() {
        ASSERT(s_Activity != NULL, "destroyed twice");
        s_Activity->myThread.join();
        delete s_Activity;
        s_Activity = NULL;
    }

    static void finish(const UserAsyncMsg& msg) {
        ASSERT(UserAsyncActivityChaining::s_Activity != NULL, "activity must be alive");
        UserAsyncActivityChaining::s_Activity->finishTaskQueue(msg);
    }

    void addWork(int addValue, int timeout = 0) {
        myQueue.push( MyTask(addValue, timeout) );
    }

    void finishTaskQueue(const UserAsyncMsg& msg) {
        myMsg = msg;
        myQueue.push( MyTask(0, 0, true) );
    }

    static int s_Result;

private:
    struct MyTask
    {
        MyTask(int addValue = 0, int timeout = 0, bool finishFlag = false)
            : myAddValue(addValue), myTimeout(timeout), myFinishFlag(finishFlag) {}

        int     myAddValue;
        int     myTimeout;
        bool    myFinishFlag;
    };

    static void threadFunc(UserAsyncActivityChaining* activity)
    {
        _TRACE_( "UserAsyncActivityChaining::threadFunc" );

        for (;;)
        {
            // Process task queue
            MyTask work;
            activity->myQueue.pop(work); // Waits until it can succeed

            _TRACE_( "UserAsyncActivityChaining::threadFunc - work: add "
                    << work.myAddValue << " (timeout = " << work.myTimeout << ")" << (work.myFinishFlag ? " FINAL" : "") );

            // 'finish flag' task is not real task, just end of queue flag
            Harness::Sleep(work.myTimeout);

            if (work.myFinishFlag) {
                break;
            }

            activity->myQueueSum += work.myAddValue;
        }

        s_Result = activity->myQueueSum;
        _TRACE_( "UserAsyncActivityChaining::threadFunc - returned result " << activity->myQueueSum );

        // Get result back to Flow Graph
        activity->myMsg.set(activity->myQueueSum);
    }

    UserAsyncActivityChaining()
        : myQueueSum(0)
        , myThread(threadFunc, this)
    {
        // Start local thread here...
        _TRACE_( "Started AsyncActivityChaining" );
    }

private: // DATA
    tbb::concurrent_bounded_queue<MyTask>   myQueue;
    int                                     myQueueSum;
    UserAsyncMsg                            myMsg;

    tbb::tbb_thread                         myThread;

    static UserAsyncActivityChaining*       s_Activity;
};

// static
UserAsyncActivityChaining* UserAsyncActivityChaining::s_Activity = NULL;
// static
int UserAsyncActivityChaining::s_Result = -4;

// override
void UserAsyncMsg::finalize() const {
    _TRACE_( "UserAsyncMsg::finalize()" );
    UserAsyncActivityChaining::finish(*this);
}

struct F3_body : tbb::internal::no_assign
{
    static int          s_FinalResult;

    int&                myI;
    bool                myAlive;

    F3_body(int& _i) : myI(_i), myAlive(true) {}

    F3_body(const F3_body& b) : myI(b.myI), myAlive(true) {}

    ~F3_body() {
        myAlive = false;
        _TRACE_( "~F3_body" );
    }

    void operator () (int result) {
        __TBB_ASSERT(myAlive, "dead node");
        // Handle async activity result here
        s_FinalResult = result;
        _TRACE_( "F3: Got async_msg result = " << result );
    }
};

// static
int F3_body::s_FinalResult = -8;

static bool testChaining() {
    bool bOk = true;
    _TRACE_( "--- SAMPLE 2 (case with chaining: F1(A<T>) ---> F2(A<T>) ---> F3(T)) " );

    for (int i = 0; i <= 2; ++i) {
        _TRACE_( "CASE " << i + 1 << ": data is " << (i > 0 ? "NOT " : "") << "ready in storage" << (i > 1 ? " NO WAITING in graph" : "") );
        _TRACE_( "MAIN THREAD" );

        tbb::flow::graph g;
        tbb::flow::function_node< tbb::flow::continue_msg, UserAsyncMsg > f1( g, tbb::flow::unlimited,
            [&]( tbb::flow::continue_msg ) -> UserAsyncMsg {
                _TRACE_( "F1: Created UserAsyncMsg" );

                UserAsyncMsg a;
                UserAsyncActivityChaining::instance()->addWork(11, (i == 0 ? 0 : 1)*ACTIVITY_PAUSE_MS_NODE1);

                return a;
            }
        );

        tbb::flow::function_node< UserAsyncMsg, UserAsyncMsg > f2( g, tbb::flow::unlimited,
            [&]( UserAsyncMsg a) -> UserAsyncMsg {
                _TRACE_( "F2: resend UserAsyncMsg" );

                UserAsyncActivityChaining::instance()->addWork(22, (i == 0 ? 0 : 1)*ACTIVITY_PAUSE_MS_NODE1);

                Harness::Sleep(ACTIVITY_PAUSE_MS_NODE2); // let activity to finish
                return a;
            }
        );

        tbb::flow::function_node< int > f3( g, tbb::flow::unlimited,
            F3_body(i)
        );

        make_edge(f1, f2);
        make_edge(f2, f3);
        f1.try_put( tbb::flow::continue_msg() );
        g.wait_for_all();

        UserAsyncActivityChaining::destroy();
        _TRACE_( "Done UserAsyncActivityChaining::destroy" );
        g.wait_for_all();
        _TRACE_( "Done g.wait_for_all()" );

        _TRACE_( "--- THE END ---" );

        if (F3_body::s_FinalResult >= 0 && UserAsyncActivityChaining::s_Result == F3_body::s_FinalResult) {
            _TRACE_( "CASE " << i + 1 << ": " << "PASSED" );
        }
        else {
            _TRACE_( "CASE " << i + 1 << ": " << "FAILED! " << UserAsyncActivityChaining::s_Result << " != " << F3_body::s_FinalResult );
            bOk = false;
            ASSERT(0, "testChaining failed");
        }
    }

    return bOk;
}

// ========================================================
namespace testFunctionsAvailabilityNS {

using namespace tbb::flow;
using tbb::flow::interface10::internal::untyped_sender;
using tbb::flow::interface10::internal::untyped_receiver;

using tbb::internal::is_same_type;
using tbb::internal::strip;
using tbb::flow::interface10::internal::wrap_tuple_elements;
using tbb::flow::interface10::internal::async_helpers;

class A {}; // Any type (usually called 'T')
struct ImpossibleType {};

template <typename T>
struct UserAsync_T   : public async_msg<T> {
    UserAsync_T() {}
    UserAsync_T(const T& t) : async_msg<T>(t) {}
};

typedef UserAsync_T<int  > UserAsync_int;
typedef UserAsync_T<float> UserAsync_float;
typedef UserAsync_T<A    > UserAsync_A;

typedef tuple< UserAsync_A, UserAsync_float, UserAsync_int, async_msg<A>, async_msg<float>, async_msg<int>, A, float, int > TypeTuple;

static int g_CheckerCounter = 0;

template <typename T, typename U>
struct CheckerTryPut {
    static ImpossibleType check( ... );

    template <typename C>
    static auto check( C* p, U* q ) -> decltype(p->try_put(*q));

    static const bool value = !is_same_type<decltype(check(static_cast<T*>(0), 0)), ImpossibleType>::value;
};

template <typename T1, typename T2>
struct CheckerMakeEdge {
    static ImpossibleType checkMake( ... );
    static ImpossibleType checkRemove( ... );

    template <typename N1, typename N2>
    static auto checkMake( N1* n1, N2* n2 ) -> decltype(tbb::flow::make_edge(*n1, *n2));

    template <typename N1, typename N2>
    static auto checkRemove( N1* n1, N2* n2 ) -> decltype(tbb::flow::remove_edge(*n1, *n2));

    static const bool valueMake   = !is_same_type<decltype(checkMake  (static_cast<T1*>(0), static_cast<T2*>(0))), ImpossibleType>::value;
    static const bool valueRemove = !is_same_type<decltype(checkRemove(static_cast<T1*>(0), static_cast<T2*>(0))), ImpossibleType>::value;

    __TBB_STATIC_ASSERT( valueMake == valueRemove, "make_edge() availability is NOT equal to remove_edge() availability" );
    
    static const bool value = valueMake;
};

template <typename T1, typename T2>
struct TypeChecker {
     TypeChecker() {
         ++g_CheckerCounter;

        REMARK("%d: %s -> %s: %s %s \n", g_CheckerCounter, typeid(T1).name(), typeid(T2).name(),
            (bAllowed ? "YES" : "no"), (bConvertable ? " (Convertable)" : ""));
     }

// 
// Check connection: function_node<continue_msg, SENDING_TYPE> <-> function_node<RECEIVING_TYPE>
//                                         R E C E I V I N G   T Y P E
// S     'bAllowed'    | int | float | A | async_msg | async_msg | async_msg | UserAsync | UserAsync | UserAsync |
// E       value       |     |       |   |   <int>   |  <float>  |    <A>    |   _int    |  _float   |   _A      |
// N   -------------------------------------------------------------------------------------------------------------
// D       int         |  Y  |       |   |     Y     |           |           |    Y      |           |           |
// I      float        |     |   Y   |   |           |    Y      |           |           |    Y      |           |
// N        A          |     |       | Y |           |           |     Y     |           |           |     Y     |
// G   async_msg<int>  |  Y  |       |   |     Y     |           |           |           |           |           |
//    async_msg<float> |     |   Y   |   |           |    Y      |           |           |           |           |
// T   async_msg<A>    |     |       | Y |           |           |     Y     |           |           |           |
// Y   UserAsync_int   |  Y  |       |   |           |           |           |    Y      |           |           |
// P  UserAsync_float  |     |   Y   |   |           |           |           |           |    Y      |           |
// E   UserAsync_A     |     |       | Y |           |           |           |           |           |    Y      |
// 
    // Test make_edge() & remove_edge() availability
    static const bool bAllowed = is_same_type<T1, T2>::value
        || is_same_type<typename async_helpers<T1>::filtered_type, T2>::value
        || is_same_type<T1, typename async_helpers<T2>::filtered_type>::value;

    static const bool bConvertable = bAllowed
        || std::is_base_of<T1, T2>::value
        || (is_same_type<typename async_helpers<T1>::filtered_type, int>::value && is_same_type<T2, float>::value)
        || (is_same_type<typename async_helpers<T1>::filtered_type, float>::value && is_same_type<T2, int>::value);

    __TBB_STATIC_ASSERT( (bAllowed == CheckerMakeEdge<function_node<continue_msg, T1>, function_node<T2> >::value), "invalid connection Fn<T1> -> Fn<T2>" );
    __TBB_STATIC_ASSERT( (bAllowed == CheckerMakeEdge<queue_node<T1>, function_node<T2> >::value), "invalid connection Queue<T1> -> Fn<T2>" );

    // Test make_edge() & remove_edge() availability with output_port<N>(node&)
    __TBB_STATIC_ASSERT( (bAllowed == CheckerMakeEdge<typename strip< decltype(
        output_port<0>( *static_cast<multifunction_node< continue_msg, tuple<T1, int> >*>(0) ) ) >::type,
        function_node<T2> >::value), "invalid connection MultuFn<0><T1,int> -> Fn<T2>" );

    __TBB_STATIC_ASSERT( (bAllowed == CheckerMakeEdge<typename strip< decltype(
        output_port<1>( *static_cast<multifunction_node< continue_msg, tuple<int, T1> >*>(0) ) ) >::type,
        function_node<T2> >::value), "invalid connection MultuFn<1><int, T1> -> Fn<T2>" );

    // Test untyped_sender connections
    __TBB_STATIC_ASSERT( (true == CheckerMakeEdge< untyped_sender, function_node<T1> >::value), "cannot connect UntypedSender -> Fn<T1>" );
    // Test untyped_receiver connections
    __TBB_STATIC_ASSERT( (true == CheckerMakeEdge< function_node<continue_msg, T1>, untyped_receiver >::value), "cannot connect F<.., T1> -> UntypedReceiver" );

    // Test untyped_receiver->try_put(T2) availability
    __TBB_STATIC_ASSERT( (true  == CheckerTryPut<untyped_receiver, T2>::value), "untyped_receiver cannot try_put(T2)" );
    // Test receiver<T1>->try_put(T2) availability
    __TBB_STATIC_ASSERT( (bConvertable == CheckerTryPut<receiver<T1>, T2>::value), "invalid availability of receiver<T1>->try_put(T2)" );
};

template <typename T1>
struct WrappedChecker {
    WrappedChecker() {} // Workaround for compilation error

    template <typename T2>
    struct T1T2Checker : TypeChecker<T1, T2> {};

    typename wrap_tuple_elements< tuple_size<TypeTuple>::value, T1T2Checker, TypeTuple >::type a;
};

typedef wrap_tuple_elements< tuple_size<TypeTuple>::value, WrappedChecker, TypeTuple >::type Checker;

} // namespace testFunctionsAvailabilityNS

static void testTryPut() {
    {
        tbb::flow::graph g;
        tbb::flow::function_node< int > f(g, tbb::flow::unlimited, [&](int) {});

        ASSERT(f.try_put(5), "try_put(int) must return true");
        ASSERT(f.try_put(7), "try_put(int) must return true");

        tbb::flow::async_msg<int> a1, a2;
        a1.set(5);
        ASSERT(f.try_put(a1), "try_put(async_msg) must return true");
        ASSERT(f.try_put(a2), "try_put(async_msg) must return true");
        a2.set(7);
        g.wait_for_all();
    }
    {
        tbb::flow::graph g;
        typedef tbb::flow::indexer_node< int >::output_type output_type;
        tbb::flow::indexer_node< int > i(g);
        tbb::flow::function_node< output_type > f(g, tbb::flow::unlimited, [&](output_type) {});
        make_edge(i, f);

        ASSERT(tbb::flow::input_port<0>(i).try_put(5), "try_put(int) must return true");
        ASSERT(tbb::flow::input_port<0>(i).try_put(7), "try_put(int) must return true");

        tbb::flow::async_msg<int> a1(5), a2(7);
        ASSERT(tbb::flow::input_port<0>(i).try_put(a1), "try_put(async_msg) must return true");
        ASSERT(tbb::flow::input_port<0>(i).try_put(a2), "try_put(async_msg) must return true");
        g.wait_for_all();
    }
}

int TestMain() {
    REMARK(" *** CHECKING FUNCTIONS: make_edge/remove_edge(node<.., T1>, node<T2>) & node<T1>->try_put(T2) ***\n");
    testFunctionsAvailabilityNS::Checker a;
    const int typeTupleSize = tbb::flow::tuple_size<testFunctionsAvailabilityNS::TypeTuple>::value;
    ASSERT(testFunctionsAvailabilityNS::g_CheckerCounter == typeTupleSize*typeTupleSize, "Type checker counter value is incorrect");

    testTryPut();

    // NOTE: Use '-v' command line argument to get traces & remarks
    tbb::task_scheduler_init init(4);
    bool bOk = true;

    for (int i = 0; i < USE_N; ++i) {
        if (i > 0 && i%1000 == 0) {
            REPORT(" *** Starting TEST %d... ***\n", i);
        }

        REMARK(" *** TEST %d ***\n", i);
        bOk = bOk && testSimplestCase();
        bOk = bOk && testChaining();
    }

    _TRACE_( " *** " << USE_N << " tests: " << (bOk ? "all tests passed" : "TESTS FAILED !!!") << " ***" );
    return (bOk ? Harness::Done : Harness::Unknown);
}

#else // __TBB_PREVIEW_ASYNC_MSG

#include "harness.h"

int TestMain() {
    return Harness::Skipped;
}

#endif // __TBB_PREVIEW_ASYNC_MSG
