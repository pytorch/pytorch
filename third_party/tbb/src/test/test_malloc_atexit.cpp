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

/* Regression test against a bug in TBB allocator manifested when
   dynamic library calls atexit() or registers dtors of static objects.
   If the allocator is not initialized yet, we can get deadlock,
   because allocator library has static object dtors as well, they
   registered during allocator initialization, and atexit() is protected
   by non-recursive mutex in some versions of GLIBC.
 */

#include <stdlib.h>
#include "harness_allocator_overload.h"

// __TBB_malloc_safer_msize() returns 0 for unknown objects,
// thus we can detect ownership
#if _USRDLL
 #if _WIN32||_WIN64
extern __declspec(dllexport)
 #endif
bool dll_isMallocOverloaded()
#else
bool exe_isMallocOverloaded()
#endif
{
    const size_t reqSz = 8;
    void *o = malloc(reqSz);
    bool ret = __TBB_malloc_safer_msize(o, NULL) >= reqSz;
    free(o);
    return ret;
}

#if _USRDLL

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"

#include <dlfcn.h>
#if __APPLE__
#include <malloc/malloc.h>
#define malloc_usable_size(p) malloc_size(p)
#else
#include <malloc.h>
#endif
#include <signal.h>

#if __linux__ && !__ANDROID__
extern "C" {
void __libc_free(void *ptr);
void *__libc_realloc(void *ptr, size_t size);

// check that such kind of free/realloc overload works correctly
void free(void *ptr)
{
    __libc_free(ptr);
}

void *realloc(void *ptr, size_t size)
{
    return __libc_realloc(ptr, size);
}
} // extern "C"
#endif // __linux__ && !__ANDROID__

#endif // MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

// Even when the test is skipped, dll source must not be empty to generate .lib to link with.

#ifndef _PGO_INSTRUMENT
void dummyFunction() {}

// TODO: enable the check under Android
#if (MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED) && !__ANDROID__
typedef void *(malloc_type)(size_t);

static void SigSegv(int)
{
    REPORT("Known issue: SIGSEGV during work with memory allocated by replaced allocator.\n"
           "skip\n");
    exit(0);
}

// TODO: Using of SIGSEGV can be eliminated via parsing /proc/self/maps
// and series of system malloc calls.
void TestReplacedAllocFunc()
{
    struct sigaction sa, sa_default;
    malloc_type *orig_malloc = (malloc_type*)dlsym(RTLD_NEXT, "malloc");
    void *p = (*orig_malloc)(16);

    // protect potentially unsafe actions
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sa.sa_handler = SigSegv;
    if (sigaction(SIGSEGV, &sa, &sa_default))
        ASSERT(0, "sigaction failed");

    ASSERT(malloc_usable_size(p) >= 16, NULL);
    free(p);
    // no more unsafe actions, restore SIGSEGV
    if (sigaction(SIGSEGV, &sa_default, NULL))
        ASSERT(0, "sigaction failed");
}
#else
void TestReplacedAllocFunc() { }
#endif

class Foo {
public:
    Foo() {
        // add a lot of exit handlers to cause memory allocation
        for (int i=0; i<1024; i++)
            atexit(dummyFunction);
        TestReplacedAllocFunc();
    }
};

static Foo f;
#endif

#else // _USRDLL
#include "harness.h"

#if _WIN32||_WIN64
#include "tbb/tbbmalloc_proxy.h"

extern __declspec(dllimport)
#endif
bool dll_isMallocOverloaded();

int TestMain () {
#ifdef _PGO_INSTRUMENT
    REPORT("Known issue: test_malloc_atexit hangs if compiled with -prof-genx\n");
    return Harness::Skipped;
#else
    ASSERT( dll_isMallocOverloaded(), "malloc was not replaced" );
    ASSERT( exe_isMallocOverloaded(), "malloc was not replaced" );
    return Harness::Done;
#endif
}

#endif // _USRDLL
