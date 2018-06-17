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

#include "tbb/tbb_config.h"

// Skip the test if no interoperability with cilkrts
#define __TBB_CILK_INTEROP   (__TBB_SURVIVE_THREAD_SWITCH && __INTEL_COMPILER>=1200)
// Skip the test when cilkrts did not have dlopen()/dlclose() start up feature
#define CILK_SYMBOLS_VISIBLE (_WIN32||_WIN64)
// The compiler does not add "-lcilkrts" linker option on some linux systems
#define CILK_LINKAGE_BROKEN  (__linux__ && __GNUC__<4 && __INTEL_COMPILER_BUILD_DATE <= 20110427)
// Currently, the interop doesn't support the situation:
//1) Intel(R) Threading Building Blocks (Intel(R) TBB) is outermost;
//2)   Intel(R) Cilk(TM) Plus, and it should be dynamically loaded with dlopen/LoadLibrary (possibly via a 3rd party module);
//3)     Intel(R) TBB again;
//4)       Intel(R) Cilk(TM) Plus again.
#define HEAVY_NESTED_INTEROP_SUPPORT ( __INTEL_COMPILER_BUILD_DATE < 20110427 )

#if __TBB_CILK_INTEROP && CILK_SYMBOLS_VISIBLE && !CILK_LINKAGE_BROKEN && HEAVY_NESTED_INTEROP_SUPPORT

#include "tbb/task_scheduler_init.h"
#include "tbb/task.h"

static const int N = 25;
static const int P_outer = 4;
static const int P_nested = 2;

#ifdef _USRDLL

#include <cilk/cilk.h>
#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"
#undef HARNESS_CUSTOM_MAIN

#if _WIN32 || _WIN64
#define CILK_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define CILK_TEST_EXPORT extern "C"
#endif /* _WIN32 || _WIN64 */

bool g_sandwich = true; // have to be declare before #include "test_cilk_common.h"
#include "test_cilk_common.h"

CILK_TEST_EXPORT int CilkFib( int n )
{
    return TBB_Fib(n);
}

CILK_TEST_EXPORT void CilkShutdown()
{
    __cilkrts_end_cilk();
}

#else /* _USRDLL undefined */

#include "harness.h"
#include "harness_dynamic_libs.h"

int SerialFib( int n ) {
    int a=0, b=1;
    for( int i=0; i<n; ++i ) {
        b += a;
        a = b-a;
    }
    return a;
}

int F = SerialFib(N);

typedef int (*CILK_CALL)(int);
CILK_CALL CilkFib = 0;

typedef void (*CILK_SHUTDOWN)();
CILK_SHUTDOWN CilkShutdown = 0;

class FibTask: public tbb::task {
    int n;
    int& result;
    task* execute() __TBB_override {
        if( n<2 ) {
            result = n;
        } else {

            // TODO: why RTLD_LAZY was used here?
            Harness::LIBRARY_HANDLE hLib =
                Harness::OpenLibrary(TEST_LIBRARY_NAME("test_cilk_dynamic_load_dll"));
            CilkFib = (CILK_CALL)Harness::GetAddress(hLib, "CilkFib");
            CilkShutdown = (CILK_SHUTDOWN)Harness::GetAddress(hLib, "CilkShutdown");

            int x, y;
            x = CilkFib(n-2);
            y = CilkFib(n-1);
            result = x+y;

            CilkShutdown();

            Harness::CloseLibrary(hLib);
        }
        return NULL;
    }
public:
    FibTask( int& result_, int n_ ) : result(result_), n(n_) {}
};


int TBB_Fib( int n ) {
    if( n<2 ) {
        return n;
    } else {
        int result;
        tbb::task_scheduler_init init(P_nested);
        tbb::task::spawn_root_and_wait(*new( tbb::task::allocate_root()) FibTask(result,n) );
        return result;
    }
}

void RunSandwich() {
    tbb::task_scheduler_init init(P_outer);
    int m = TBB_Fib(N);
    ASSERT( m == F, NULL );
}

int TestMain () {
    for ( int i = 0; i < 20; ++i )
        RunSandwich();
    return Harness::Done;
}

#endif /* _USRDLL */

#else /* !__TBB_CILK_INTEROP */

#include "harness.h"

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__TBB_CILK_INTEROP */
