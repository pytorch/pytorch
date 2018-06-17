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


#if (_WIN32 || _WIN64)
// As the test is intentionally build with /EHs-, suppress multiple VS2005's
// warnings like C4530: C++ exception handler used, but unwind semantics are not enabled
#if defined(_MSC_VER) && !__INTEL_COMPILER
/* ICC 10.1 and 11.0 generates code that uses std::_Raise_handler,
   but it's only defined in libcpmt(d), which the test doesn't linked with.
 */
#undef  _HAS_EXCEPTIONS
#define _HAS_EXCEPTIONS _CPPUNWIND
#endif
// to use strdup w/o warnings
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif // _WIN32 || _WIN64

#define _ISOC11_SOURCE 1 // to get C11 declarations for GLIBC
#define HARNESS_NO_PARSE_COMMAND_LINE 1

#include "harness_allocator_overload.h"

#if MALLOC_WINDOWS_OVERLOAD_ENABLED
#include "tbb/tbbmalloc_proxy.h"
#endif

#include "harness.h"

#if !HARNESS_SKIP_TEST

#if __ANDROID__
  #include <android/api-level.h> // for __ANDROID_API__
#endif

#define __TBB_POSIX_MEMALIGN_PRESENT (__linux__ && !__ANDROID__) || __APPLE__
#define __TBB_PVALLOC_PRESENT __linux__ && !__ANDROID__
#if __GLIBC__
  // aligned_alloc available since GLIBC 2.16
  #define __TBB_ALIGNED_ALLOC_PRESENT __GLIBC_PREREQ(2, 16)
#endif // __GLIBC__
 // later Android doesn't have valloc or dlmalloc_usable_size
#define __TBB_VALLOC_PRESENT (__linux__ && __ANDROID_API__<21) || __APPLE__
#define __TBB_DLMALLOC_USABLE_SIZE_PRESENT  __ANDROID__ && __ANDROID_API__<21

#include "harness_report.h"
#include "harness_assert.h"
#include <stdlib.h>
#include <string.h>
#if !__APPLE__
#include <malloc.h>
#endif
#include <stdio.h>
#include <new>
#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED
#include <unistd.h> // for sysconf
#include <dlfcn.h>
#endif

#if __linux__
#include <stdint.h> // for uintptr_t

extern "C" {
void *__libc_malloc(size_t size);
void *__libc_realloc(void *ptr, size_t size);
void *__libc_calloc(size_t num, size_t size);
void __libc_free(void *ptr);
void *__libc_memalign(size_t alignment, size_t size);
void *__libc_pvalloc(size_t size);
void *__libc_valloc(size_t size);
#if __TBB_DLMALLOC_USABLE_SIZE_PRESENT
#define malloc_usable_size(p) dlmalloc_usable_size(p)
size_t dlmalloc_usable_size(const void *ptr);
#endif
}

#elif __APPLE__

#include <malloc/malloc.h>
#define malloc_usable_size(p) malloc_size(p)

#elif _WIN32
#include <stddef.h>
#if __MINGW32__
#include <unistd.h>
#else
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif

#endif /* OS selection */

#if _WIN32
// On Windows, the trick with string "dependency on msvcpXX.dll" is necessary to create
// dependency on msvcpXX.dll, for sake of a regression test.
// On Linux, C++ RTL headers are undesirable because of breaking strict ANSI mode.
#if defined(_MSC_VER) && _MSC_VER >= 1300 && _MSC_VER <= 1310 && !defined(__INTEL_COMPILER)
/* Fixing compilation error reported by VS2003 for exception class
   when _HAS_EXCEPTIONS is 0:
   bad_cast that inherited from exception is not in std namespace.
*/
using namespace std;
#endif
#include <string>
#endif

#include "../tbbmalloc/shared_utils.h"  // alignDown, alignUp, estimatedCacheLineSize

/* start of code replicated from src/tbbmalloc */

class BackRefIdx { // composite index to backreference array
private:
    uint16_t master;      // index in BackRefMaster
    uint16_t largeObj:1;  // is this object "large"?
    uint16_t offset  :15; // offset from beginning of BackRefBlock
public:
    BackRefIdx() : master((uint16_t)-1) {}
    bool isInvalid() { return master == (uint16_t)-1; }
    bool isLargeObject() const { return largeObj; }
    uint16_t getMaster() const { return master; }
    uint16_t getOffset() const { return offset; }

    // only newBackRef can modify BackRefIdx
    static BackRefIdx newBackRef(bool largeObj);
};

class MemoryPool;
class ExtMemoryPool;

class BlockI {
    intptr_t     blockState[2];
};

struct LargeMemoryBlock : public BlockI {
    MemoryPool       *pool;          // owner pool
    LargeMemoryBlock *next,          // ptrs in list of cached blocks
                     *prev,
                     *gPrev,         // in pool's global list
                     *gNext;
    uintptr_t         age;           // age of block while in cache
    size_t            objectSize;    // the size requested by a client
    size_t            unalignedSize; // the size requested from getMemory
    bool              fromMapMemory;
    BackRefIdx        backRefIdx;    // cached here, used copy is in LargeObjectHdr
    void registerInPool(ExtMemoryPool *extMemPool);
    void unregisterFromPool(ExtMemoryPool *extMemPool);
};

struct LargeObjectHdr {
    LargeMemoryBlock *memoryBlock;
    /* Have to duplicate it here from CachedObjectHdr,
       as backreference must be checked without further pointer dereference.
       Points to LargeObjectHdr. */
    BackRefIdx       backRefIdx;
};

/*
 * Objects of size minLargeObjectSize and larger are considered large objects.
 */
const uintptr_t blockSize = 16*1024;
const uint32_t fittingAlignment = rml::internal::estimatedCacheLineSize;
#define SET_FITTING_SIZE(N) ( (blockSize-2*rml::internal::estimatedCacheLineSize)/N ) & ~(fittingAlignment-1)
const uint32_t fittingSize5 = SET_FITTING_SIZE(2); // 8128/8064
#undef SET_FITTING_SIZE
const uint32_t minLargeObjectSize = fittingSize5 + 1;

/* end of code replicated from src/tbbmalloc */

static void scalableMallocCheckSize(void *object, size_t size)
{
#if __clang__
// This prevents Clang from throwing out the calls to new & delete in CheckNewDeleteOverload().
    static void *v = object;
#endif
    ASSERT(object, NULL);
    if (size >= minLargeObjectSize) {
        LargeMemoryBlock *lmb = ((LargeObjectHdr*)object-1)->memoryBlock;
        ASSERT(uintptr_t(lmb)<uintptr_t(((LargeObjectHdr*)object-1))
               && lmb->objectSize >= size, NULL);
    }
#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED
    ASSERT(malloc_usable_size(object) >= size, NULL);
#elif MALLOC_WINDOWS_OVERLOAD_ENABLED
    // Check that _msize works correctly
    ASSERT(_msize(object) >= size, NULL);
    ASSERT(size<8 || _aligned_msize(object,8,0) >= size, NULL);
#endif
}

void CheckStdFuncOverload(void *(*malloc_p)(size_t), void *(*calloc_p)(size_t, size_t),
                          void *(*realloc_p)(void *, size_t), void (*free_p)(void *))
{
    void *ptr = malloc_p(minLargeObjectSize);
    scalableMallocCheckSize(ptr, minLargeObjectSize);
    free(ptr);

    ptr = calloc_p(minLargeObjectSize, 2);
    scalableMallocCheckSize(ptr, 2*minLargeObjectSize);
    void *ptr1 = realloc_p(ptr, 10*minLargeObjectSize);
    scalableMallocCheckSize(ptr1, 10*minLargeObjectSize);
    free_p(ptr1);
}

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

void CheckMemalignFuncOverload(void *(*memalign_p)(size_t, size_t),
                               void (*free_p)(void*))
{
    void *ptr = memalign_p(128, 4*minLargeObjectSize);
    scalableMallocCheckSize(ptr, 4*minLargeObjectSize);
    ASSERT(is_aligned(ptr, 128), NULL);
    free_p(ptr);
}

void CheckVallocFuncOverload(void *(*valloc_p)(size_t), void (*free_p)(void*))
{
    void *ptr = valloc_p(minLargeObjectSize);
    scalableMallocCheckSize(ptr, minLargeObjectSize);
    ASSERT(is_aligned(ptr, sysconf(_SC_PAGESIZE)), NULL);
    free_p(ptr);
}

void CheckPvalloc(void *(*pvalloc_p)(size_t), void (*free_p)(void*))
{
    const long memoryPageSize = sysconf(_SC_PAGESIZE);
    // request large object with not power-of-2 size
    const size_t largeSz = alignUp(minLargeObjectSize, 16*1024) + 1;

    for (size_t sz = 0; sz<=largeSz; sz+=largeSz) {
        void *ptr = pvalloc_p(sz);
        scalableMallocCheckSize(ptr, sz? alignUp(sz, memoryPageSize) : memoryPageSize);
        ASSERT(is_aligned(ptr, memoryPageSize), NULL);
        free_p(ptr);
    }
}

#endif // MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

// regression test: on macOS scalable_free() treated small aligned object,
// placed in large block, as small block
void CheckFreeAligned() {
    size_t sz[] = {8, 4*1024, 16*1024, 0};
    size_t align[] = {8, 4*1024, 16*1024, 0};

    for (int s=0; sz[s]; s++)
        for (int a=0; align[a]; a++) {
            void *ptr = NULL;
#if __TBB_POSIX_MEMALIGN_PRESENT
            int ret = posix_memalign(&ptr, align[a], sz[s]);
            ASSERT(!ret, NULL);
#elif MALLOC_WINDOWS_OVERLOAD_ENABLED
            ptr = _aligned_malloc(sz[s], align[a]);
#endif
            ASSERT(is_aligned(ptr, align[a]), NULL);
            free(ptr);
        }
}

#if __ANDROID__
// Workaround for an issue with strdup somehow bypassing our malloc replacement on Android.
char *strdup(const char *str) {
    REPORT( "Known issue: malloc replacement does not work for strdup on Android.\n" );
    size_t len = strlen(str)+1;
    void *new_str = malloc(len);
    return new_str ? reinterpret_cast<char *>(memcpy(new_str, str, len)) : 0;
}
#endif

#if __APPLE__
#include <mach/mach.h>

// regression test: malloc_usable_size() that was passed to zone interface
// called system malloc_usable_size(), so for object that was not allocated
// by tbbmalloc non-zero was returned, so such objects were passed to
// tbbmalloc's free(), that is incorrect
void TestZoneOverload() {
    vm_address_t *zones;
    unsigned zones_num;

    kern_return_t ret = malloc_get_all_zones(mach_task_self(), NULL, &zones, &zones_num);
    ASSERT(!ret && zones_num>1, NULL);
    malloc_zone_t *sys_zone = (malloc_zone_t*)zones[1];
    ASSERT(strcmp("tbbmalloc", malloc_get_zone_name(sys_zone)),
                  "zone 1 expected to be not tbbmalloc");
    void *p = malloc_zone_malloc(sys_zone, 16);
    free(p);
}
#else
#define TestZoneOverload()
#endif

#if _WIN32
// regression test: certain MSVC runtime functions use "public" allocation functions
// but internal free routines, causing crashes if tbbmalloc_proxy does not intercept the latter.
void TestRuntimeRoutines() {
    system("rem should be a safe command to call");
}
#else
#define TestRuntimeRoutines()
#endif

struct BigStruct {
    char f[minLargeObjectSize];
};

void CheckNewDeleteOverload() {
    BigStruct *s1, *s2, *s3, *s4;

    s1 = new BigStruct;
    scalableMallocCheckSize(s1, sizeof(BigStruct));
    delete s1;

    s2 = new BigStruct[10];
    scalableMallocCheckSize(s2, 10*sizeof(BigStruct));
    delete []s2;

    s3 = new(std::nothrow) BigStruct;
    scalableMallocCheckSize(s3, sizeof(BigStruct));
    delete s3;

    s4 = new(std::nothrow) BigStruct[2];
    scalableMallocCheckSize(s4, 2*sizeof(BigStruct));
    delete []s4;
}

int TestMain() {
    void *ptr, *ptr1;

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED
    ASSERT(dlsym(RTLD_DEFAULT, "scalable_malloc"),
           "Lost dependency on malloc_proxy or LD_PRELOAD was not set?");
#endif

/* On Windows, memory block size returned by _msize() is sometimes used
   to calculate the size for an extended block. Substituting _msize,
   scalable_msize initially returned 0 for regions not allocated by the scalable
   allocator, which led to incorrect memory reallocation and subsequent crashes.
   It was found that adding a new environment variable triggers the error.
*/
    ASSERT(getenv("PATH"), "We assume that PATH is set everywhere.");
    char *pathCopy = strdup(getenv("PATH"));
#if __ANDROID__
    ASSERT(strcmp(pathCopy,getenv("PATH")) == 0, "strdup workaround does not work as expected.");
#endif
    const char *newEnvName = "__TBBMALLOC_OVERLOAD_REGRESSION_TEST_FOR_REALLOC_AND_MSIZE";
    ASSERT(!getenv(newEnvName), "Environment variable should not be used before.");
    int r = Harness::SetEnv(newEnvName,"1");
    ASSERT(!r, NULL);
    char *path = getenv("PATH");
    ASSERT(path && 0==strcmp(path, pathCopy), "Environment was changed erroneously.");
    free(pathCopy);

    CheckStdFuncOverload(malloc, calloc, realloc, free);
#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

#if __TBB_POSIX_MEMALIGN_PRESENT
    int ret = posix_memalign(&ptr, 1024, 3*minLargeObjectSize);
    scalableMallocCheckSize(ptr, 3*minLargeObjectSize);
    ASSERT(0==ret && is_aligned(ptr, 1024), NULL);
    free(ptr);
#endif

#if __TBB_VALLOC_PRESENT
    CheckVallocFuncOverload(valloc, free);
#endif
#if __TBB_PVALLOC_PRESENT
    CheckPvalloc(pvalloc, free);
#endif
#if __linux__
    CheckMemalignFuncOverload(memalign, free);
#if __TBB_ALIGNED_ALLOC_PRESENT
    CheckMemalignFuncOverload(aligned_alloc, free);
#endif

    struct mallinfo info = mallinfo();
    // right now mallinfo initialized by zero
    ASSERT(!info.arena && !info.ordblks && !info.smblks && !info.hblks
           && !info.hblkhd && !info.usmblks && !info.fsmblks
           && !info.uordblks && !info.fordblks && !info.keepcost, NULL);

 #if !__ANDROID__
    // These non-standard functions are exported by GLIBC, and might be used
    // in conjunction with standard malloc/free. Test that we overload them as well.
    // Bionic doesn't have them.
    CheckStdFuncOverload(__libc_malloc, __libc_calloc, __libc_realloc, __libc_free);
    CheckMemalignFuncOverload(__libc_memalign, __libc_free);
    CheckVallocFuncOverload(__libc_valloc, __libc_free);
    CheckPvalloc(__libc_pvalloc, __libc_free);
 #endif
#endif // __linux__

#else // MALLOC_WINDOWS_OVERLOAD_ENABLED

    ptr = _aligned_malloc(minLargeObjectSize, 16);
    scalableMallocCheckSize(ptr, minLargeObjectSize);
    ASSERT(is_aligned(ptr, 16), NULL);

    // Testing of workaround for vs "is power of 2 pow N" bug that accepts zeros
    ptr1 = _aligned_malloc(minLargeObjectSize, 0);
    scalableMallocCheckSize(ptr, minLargeObjectSize);
    ASSERT(is_aligned(ptr, sizeof(void*)), NULL);
    _aligned_free(ptr1);

    ptr1 = _aligned_realloc(ptr, minLargeObjectSize*10, 16);
    scalableMallocCheckSize(ptr1, minLargeObjectSize*10);
    ASSERT(is_aligned(ptr, 16), NULL);
    _aligned_free(ptr1);

#endif
    CheckFreeAligned();

    CheckNewDeleteOverload();
#if _WIN32
    std::string stdstring = "dependency on msvcpXX.dll";
    ASSERT(strcmp(stdstring.c_str(), "dependency on msvcpXX.dll") == 0, NULL);
#endif
    TestZoneOverload();
    TestRuntimeRoutines();

    return Harness::Done;
}
#endif // !HARNESS_SKIP_TEST
