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

#if _USRDLL

#include <stdlib.h> // for NULL
#include "harness_assert.h"
#define HARNESS_CUSTOM_MAIN 1
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

const char *globalCallMsg = "A TBB allocator function call is resolved into wrong implementation.";

#if _WIN32||_WIN64
// must be defined in DLL for linker to not drop the dependency on the DLL.
extern "C" {
    extern __declspec(dllexport) void *scalable_malloc(size_t);
    extern __declspec(dllexport) void scalable_free (void *);
    extern __declspec(dllexport) void safer_scalable_free (void *, void (*)(void*));
    extern __declspec(dllexport) void *scalable_realloc(void *, size_t);
    extern __declspec(dllexport) void *safer_scalable_realloc(void *, size_t, void *);
    extern __declspec(dllexport) void *scalable_calloc(size_t, size_t);
    extern __declspec(dllexport) int scalable_posix_memalign(void **, size_t, size_t);
    extern __declspec(dllexport) void *scalable_aligned_malloc(size_t, size_t);
    extern __declspec(dllexport) void *scalable_aligned_realloc(void *, size_t, size_t);
    extern __declspec(dllexport) void *safer_scalable_aligned_realloc(void *, size_t, size_t, void *);
    extern __declspec(dllexport) void scalable_aligned_free(void *);
    extern __declspec(dllexport) size_t scalable_msize(void *);
    extern __declspec(dllexport) size_t safer_scalable_msize (void *, size_t (*)(void*));
}
#endif

// Those functions must not be called instead of presented in dynamic library.
extern "C" void *scalable_malloc(size_t)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void scalable_free (void *)
{
    ASSERT(0, globalCallMsg);
}
extern "C" void safer_scalable_free (void *, void (*)(void*))
{
    ASSERT(0, globalCallMsg);
}
extern "C" void *scalable_realloc(void *, size_t)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void *safer_scalable_realloc(void *, size_t, void *)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void *scalable_calloc(size_t, size_t)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" int scalable_posix_memalign(void **, size_t, size_t)
{
    ASSERT(0, globalCallMsg);
    return 0;
}
extern "C" void *scalable_aligned_malloc(size_t, size_t)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void *scalable_aligned_realloc(void *, size_t, size_t)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void *safer_scalable_aligned_realloc(void *, size_t, size_t, void *)
{
    ASSERT(0, globalCallMsg);
    return NULL;
}
extern "C" void scalable_aligned_free(void *)
{
    ASSERT(0, globalCallMsg);
}
extern "C" size_t scalable_msize(void *)
{
    ASSERT(0, globalCallMsg);
    return 0;
}
extern "C" size_t safer_scalable_msize (void *, size_t (*)(void*))
{
    ASSERT(0, globalCallMsg);
    return 0;
}

#else  // _USRDLL

// harness_defs.h must be included before tbb_stddef.h to overcome exception-dependent
// system headers that come from tbb_stddef.h
#include "harness_defs.h"
#include "tbb/tbb_stddef.h"
#if __TBB_WIN8UI_SUPPORT || __TBB_SOURCE_DIRECTLY_INCLUDED || __TBB_MIC_OFFLOAD
// The test does not work if dynamic load is unavailable.
// For MIC offload, it fails because liboffload brings libiomp which observes and uses the fake scalable_* calls.
#define HARNESS_SKIP_TEST 1
#endif
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#define HARNESS_TBBMALLOC_THREAD_SHUTDOWN 1
#include "harness.h"

#if !HARNESS_SKIP_TEST

#include "harness_dynamic_libs.h"
#include "harness_memory.h"

extern "C" {
#if _WIN32||_WIN64
extern __declspec(dllimport)
#endif
void *scalable_malloc(size_t);
}

struct Run {
    void operator()( int /*id*/ ) const {
        using namespace Harness;

        void* (*malloc_ptr)(std::size_t);
        void (*free_ptr)(void*);

        void* (*aligned_malloc_ptr)(size_t size, size_t alignment);
        void  (*aligned_free_ptr)(void*);

        const char* actual_name;
        LIBRARY_HANDLE lib = OpenLibrary(actual_name = MALLOCLIB_NAME1);
        if (!lib)      lib = OpenLibrary(actual_name = MALLOCLIB_NAME2);
        if (!lib) {
            REPORT("Can't load " MALLOCLIB_NAME1 " or " MALLOCLIB_NAME2 "\n");
            exit(1);
        }
        (FunctionAddress&)malloc_ptr = GetAddress(lib, "scalable_malloc");
        (FunctionAddress&)free_ptr = GetAddress(lib, "scalable_free");
        (FunctionAddress&)aligned_malloc_ptr = GetAddress(lib, "scalable_aligned_malloc");
        (FunctionAddress&)aligned_free_ptr = GetAddress(lib, "scalable_aligned_free");

        for (size_t sz = 1024; sz <= 10*1024 ; sz*=10) {
            void *p1 = aligned_malloc_ptr(sz, 16);
            memset(p1, 0, sz);
            aligned_free_ptr(p1);
        }

        void *p = malloc_ptr(100);
        memset(p, 1, 100);
        free_ptr(p);

        CloseLibrary(lib);
#if _WIN32 || _WIN64
        ASSERT(GetModuleHandle(actual_name),
               "allocator library must not be unloaded");
#else
        ASSERT(dlsym(RTLD_DEFAULT, "scalable_malloc"),
               "allocator library must not be unloaded");
#endif
    }
};

int TestMain () {
    int i;
    std::ptrdiff_t memory_leak;

    // warm-up run
    NativeParallelFor( 1, Run() );
    /* 1st call to GetMemoryUsage() allocate some memory,
       but it seems memory consumption stabilized after this.
     */
    GetMemoryUsage();
    std::size_t memory_in_use = GetMemoryUsage();
    ASSERT(memory_in_use == GetMemoryUsage(),
           "Memory consumption should not increase after 1st GetMemoryUsage() call");

    // expect that memory consumption stabilized after several runs
    for (i=0; i<3; i++) {
        std::size_t memory_in_use = GetMemoryUsage();
        for (int j=0; j<10; j++)
            NativeParallelFor( 1, Run() );
        memory_leak = GetMemoryUsage() - memory_in_use;
        if (memory_leak == 0)  // possibly too strong?
            break;
    }
    if(3==i) {
        // not stabilized, could be leak
        REPORT( "Error: memory leak of up to %ld bytes\n", static_cast<long>(memory_leak));
        exit(1);
    }

    return Harness::Done;
}

#endif /* HARNESS_SKIP_TEST */

#endif // _USRDLL
