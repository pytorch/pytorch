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

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(disable: 4180) // "qualifier applied to function type has no meaning; ignored"
#endif

#ifndef TBB_PREVIEW_VARIADIC_PARALLEL_INVOKE
    #define TBB_PREVIEW_VARIADIC_PARALLEL_INVOKE __TBB_CPF_BUILD
#endif

#include "tbb/parallel_invoke.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"
#include "tbb/tbb_exception.h"
#include "harness.h"

#if !__INTEL_COMPILER && (_MSC_VER && _MSC_VER <= 1400 || __GNUC__==3 && __GNUC_MINOR__<=3 || __SUNPRO_CC)
    #define __TBB_FUNCTION_BY_CONSTREF_IN_TEMPLATE_BROKEN 1
#endif

tbb::atomic<size_t> function_counter;

// Some macros to make the test easier to read

// 10 functions test0 ... test9 are defined
// pointer to each function is also defined

#define TEST_FUNCTION(value) void test##value () \
{   \
    ASSERT(!(function_counter & (1 << value)), "Test function has already been called"); \
    function_counter += 1 << value; \
}   \
void (*test_pointer##value)(void) = test##value;

TEST_FUNCTION(0)
TEST_FUNCTION(1)
TEST_FUNCTION(2)
TEST_FUNCTION(3)
TEST_FUNCTION(4)
TEST_FUNCTION(5)
TEST_FUNCTION(6)
TEST_FUNCTION(7)
TEST_FUNCTION(8)
TEST_FUNCTION(9)

// The same with functors
#define TEST_FUNCTOR(value) class test_functor##value  \
{   \
public: \
    void operator() () const {  \
        function_counter += 1 << value;   \
    }   \
} functor##value;

TEST_FUNCTOR(0)
TEST_FUNCTOR(1)
TEST_FUNCTOR(2)
TEST_FUNCTOR(3)
TEST_FUNCTOR(4)
TEST_FUNCTOR(5)
TEST_FUNCTOR(6)
TEST_FUNCTOR(7)
TEST_FUNCTOR(8)
TEST_FUNCTOR(9)

#define INIT_TEST function_counter = 0;

#define VALIDATE_INVOKE_RUN(number_of_args, test_type) \
    ASSERT( size_t(function_counter) == (size_t(1) << number_of_args) - 1, "parallel_invoke called with " #number_of_args " arguments didn't process all " #test_type);

// Calls parallel_invoke for different number of arguments
// It can be called with and without user context
template <typename F0, typename F1, typename F2, typename F3, typename F4, typename F5,
    typename F6, typename F7, typename F8, typename F9>
void call_parallel_invoke( size_t n, F0& f0, F1& f1, F2& f2, F3& f3, F4 &f4, F5 &f5,
                          F6& f6, F7 &f7, F8 &f8, F9 &f9, tbb::task_group_context* context) {
    switch(n) {
    default:
        ASSERT(false, "number of arguments must be between 2 and 10");
    case 2:
        if (context)
            tbb::parallel_invoke (f0, f1, *context);
        else
            tbb::parallel_invoke (f0, f1);
        break;
    case 3:
        if (context)
            tbb::parallel_invoke (f0, f1, f2, *context);
        else
            tbb::parallel_invoke (f0, f1, f2);
        break;
    case 4:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3);
        break;
    case 5:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4);
        break;
    case 6:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5);
        break;
    case 7:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6);
        break;
    case 8:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7);
        break;
    case 9:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7, f8, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7, f8);
        break;
    case 10:
        if(context)
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, *context);
        else
            tbb::parallel_invoke (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
        break;
    }
}

#if !__TBB_FUNCTION_BY_CONSTREF_IN_TEMPLATE_BROKEN
template<typename function> void aux_invoke(const function& f) {
    f();
}

bool function_by_constref_in_template_codegen_broken() {
    function_counter = 0;
    aux_invoke(test1);
    return function_counter==0;
}
#endif /* !__TBB_FUNCTION_BY_CONSTREF_IN_TEMPLATE_BROKEN */

void test_parallel_invoke()
{
    REMARK (__FUNCTION__);
    // Testing with pointers to functions
    for (int n = 2; n <=10; n++)
    {
        INIT_TEST;
        call_parallel_invoke(n, test_pointer0, test_pointer1, test_pointer2, test_pointer3, test_pointer4,
            test_pointer5, test_pointer6, test_pointer7, test_pointer8, test_pointer9, NULL);
        VALIDATE_INVOKE_RUN(n, "pointers to function");
    }

    // Testing parallel_invoke with functors
    for (int n = 2; n <=10; n++)
    {
        INIT_TEST;
        call_parallel_invoke(n, functor0, functor1, functor2, functor3, functor4,
            functor5, functor6, functor7, functor8, functor9, NULL);
        VALIDATE_INVOKE_RUN(n, "functors");
    }

#if __TBB_FUNCTION_BY_CONSTREF_IN_TEMPLATE_BROKEN
    // some old compilers can't cope with passing function name into parallel_invoke
#else
    // and some compile but generate broken code that does not call the function
    if (function_by_constref_in_template_codegen_broken())
        return;

    // Testing parallel_invoke with functions
    for (int n = 2; n <=10; n++)
    {
        INIT_TEST;
        call_parallel_invoke(n, test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, NULL);
        VALIDATE_INVOKE_RUN(n, "functions");
    }
#endif
}

// Exception handling support test

#if __TBB_TASK_GROUP_CONTEXT
#define HARNESS_EH_SIMPLE_MODE 1
#include "harness_eh.h"

#if TBB_USE_EXCEPTIONS
volatile size_t exception_mask; // each bit represents whether the function should throw exception or not

// throws exception if corresponding exception_mask bit is set
#define TEST_FUNCTOR_WITH_THROW(value) \
struct throwing_functor##value { \
    void operator() () const {  \
        if (exception_mask & (1 << value))   \
            ThrowTestException();    \
    }   \
} test_with_throw##value;

TEST_FUNCTOR_WITH_THROW(0)
TEST_FUNCTOR_WITH_THROW(1)
TEST_FUNCTOR_WITH_THROW(2)
TEST_FUNCTOR_WITH_THROW(3)
TEST_FUNCTOR_WITH_THROW(4)
TEST_FUNCTOR_WITH_THROW(5)
TEST_FUNCTOR_WITH_THROW(6)
TEST_FUNCTOR_WITH_THROW(7)
TEST_FUNCTOR_WITH_THROW(8)
TEST_FUNCTOR_WITH_THROW(9)

void TestExceptionHandling()
{
    REMARK (__FUNCTION__);
    for( size_t n = 2; n <= 10; ++n ) {
        for( exception_mask = 1; exception_mask < (size_t(1) << n); ++exception_mask ) {
            ResetEhGlobals();
            TRY();
                REMARK("Calling parallel_invoke, number of functions = %d, exception_mask = %d\n", n, exception_mask);
                call_parallel_invoke(n, test_with_throw0, test_with_throw1, test_with_throw2, test_with_throw3,
                    test_with_throw4, test_with_throw5, test_with_throw6, test_with_throw7, test_with_throw8, test_with_throw9, NULL);
            CATCH_AND_ASSERT();
        }
    }
}
#endif /* TBB_USE_EXCEPTIONS */

// Cancellation support test
void function_to_cancel() {
    ++g_CurExecuted;
    CancellatorTask::WaitUntilReady();
}

// The function is used to test cancellation
void simple_test_nothrow (){
    ++g_CurExecuted;
}

size_t g_numFunctions,
       g_functionToCancel;

class ParInvokeLauncherTask : public tbb::task
{
    tbb::task_group_context &my_ctx;
    void(*func_array[10])(void);

    tbb::task* execute () __TBB_override {
        func_array[g_functionToCancel] = &function_to_cancel;
        call_parallel_invoke(g_numFunctions, func_array[0], func_array[1], func_array[2], func_array[3],
            func_array[4], func_array[5], func_array[6], func_array[7], func_array[8], func_array[9], &my_ctx);
        return NULL;
    }
public:
    ParInvokeLauncherTask ( tbb::task_group_context& ctx ) : my_ctx(ctx) {
        for (int i = 0; i <=9; ++i)
            func_array[i] = &simple_test_nothrow;
    }
};

void TestCancellation ()
{
    REMARK (__FUNCTION__);
    for ( int n = 2; n <= 10; ++n ) {
        for ( int m = 0; m <= n - 1; ++m ) {
            g_numFunctions = n;
            g_functionToCancel = m;
            ResetEhGlobals();
            RunCancellationTest<ParInvokeLauncherTask, CancellatorTask>();
        }
    }
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

//------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------

#include "harness_cpu.h"

int TestMain () {
    MinThread = min(MinThread, MaxThread);
    ASSERT (MinThread>=1, "Minimal number of threads must be 1 or more");
    for ( int p = MinThread; p <= MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        test_parallel_invoke();
        if (p > 1) {
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
            REPORT("Known issue: exception handling tests are skipped.\n");
#elif TBB_USE_EXCEPTIONS
            TestExceptionHandling();
#endif /* TBB_USE_EXCEPTIONS */
#if __TBB_TASK_GROUP_CONTEXT
            TestCancellation();
#endif /* __TBB_TASK_GROUP_CONTEXT */
        }
        TestCPUUserTime(p);
    }
    return Harness::Done;
}
