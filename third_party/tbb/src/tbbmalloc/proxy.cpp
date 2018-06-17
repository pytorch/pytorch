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

#include "proxy.h"
#include "tbb/tbb_config.h"

#if !defined(__EXCEPTIONS) && !defined(_CPPUNWIND) && !defined(__SUNPRO_CC)
    #if TBB_USE_EXCEPTIONS
        #error Compilation settings do not support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
    #elif !defined(TBB_USE_EXCEPTIONS)
        #define TBB_USE_EXCEPTIONS 0
    #endif
#elif !defined(TBB_USE_EXCEPTIONS)
    #define TBB_USE_EXCEPTIONS 1
#endif

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED

#ifndef __THROW
#define __THROW
#endif

/*** service functions and variables ***/

#include <string.h> // for memset
#include <unistd.h> // for sysconf

static long memoryPageSize;

static inline void initPageSize()
{
    memoryPageSize = sysconf(_SC_PAGESIZE);
}

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#include "Customize.h" // FencedStore
#include <dlfcn.h>
#include <malloc.h>    // mallinfo

/* __TBB_malloc_proxy used as a weak symbol by libtbbmalloc for:
   1) detection that the proxy library is loaded
   2) check that dlsym("malloc") found something different from our replacement malloc
*/
extern "C" void *__TBB_malloc_proxy(size_t) __attribute__ ((alias ("malloc")));

static void *orig_msize;

#elif MALLOC_ZONE_OVERLOAD_ENABLED

#include "proxy_overload_osx.h"

#endif // MALLOC_ZONE_OVERLOAD_ENABLED

// Original (i.e., replaced) functions,
// they are never changed for MALLOC_ZONE_OVERLOAD_ENABLED.
static void *orig_free,
    *orig_realloc;

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#define ZONE_ARG
#define PREFIX(name) name

static void *orig_libc_free,
    *orig_libc_realloc;

// We already tried to find ptr to original functions.
static intptr_t origFuncSearched;

inline void InitOrigPointers()
{
    // race is OK here, as different threads found same functions
    if (!origFuncSearched) {
        orig_free = dlsym(RTLD_NEXT, "free");
        orig_realloc = dlsym(RTLD_NEXT, "realloc");
        orig_msize = dlsym(RTLD_NEXT, "malloc_usable_size");
        orig_libc_free = dlsym(RTLD_NEXT, "__libc_free");
        orig_libc_realloc = dlsym(RTLD_NEXT, "__libc_realloc");

        FencedStore(origFuncSearched, 1);
    }
}

/*** replacements for malloc and the family ***/
extern "C" {
#elif MALLOC_ZONE_OVERLOAD_ENABLED

// each impl_* function has such 1st argument, it's unused
#define ZONE_ARG struct _malloc_zone_t *,
#define PREFIX(name) impl_##name
// not interested in original functions for zone overload
inline void InitOrigPointers() {}

#endif // MALLOC_UNIXLIKE_OVERLOAD_ENABLED and MALLOC_ZONE_OVERLOAD_ENABLED

void *PREFIX(malloc)(ZONE_ARG size_t size) __THROW
{
    return scalable_malloc(size);
}

void *PREFIX(calloc)(ZONE_ARG size_t num, size_t size) __THROW
{
    return scalable_calloc(num, size);
}

void PREFIX(free)(ZONE_ARG void *object) __THROW
{
    InitOrigPointers();
    __TBB_malloc_safer_free(object, (void (*)(void*))orig_free);
}

void *PREFIX(realloc)(ZONE_ARG void* ptr, size_t sz) __THROW
{
    InitOrigPointers();
    return __TBB_malloc_safer_realloc(ptr, sz, orig_realloc);
}

/* The older *NIX interface for aligned allocations;
   it's formally substituted by posix_memalign and deprecated,
   so we do not expect it to cause cyclic dependency with C RTL. */
void *PREFIX(memalign)(ZONE_ARG size_t alignment, size_t size) __THROW
{
    return scalable_aligned_malloc(size, alignment);
}

/* valloc allocates memory aligned on a page boundary */
void *PREFIX(valloc)(ZONE_ARG size_t size) __THROW
{
    if (! memoryPageSize) initPageSize();

    return scalable_aligned_malloc(size, memoryPageSize);
}

#undef ZONE_ARG
#undef PREFIX

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED

// match prototype from system headers
#if __ANDROID__
size_t malloc_usable_size(const void *ptr) __THROW
#else
size_t malloc_usable_size(void *ptr) __THROW
#endif
{
    InitOrigPointers();
    return __TBB_malloc_safer_msize(const_cast<void*>(ptr), (size_t (*)(void*))orig_msize);
}

int posix_memalign(void **memptr, size_t alignment, size_t size) __THROW
{
    return scalable_posix_memalign(memptr, alignment, size);
}

/* pvalloc allocates smallest set of complete pages which can hold
   the requested number of bytes. Result is aligned on page boundary. */
void *pvalloc(size_t size) __THROW
{
    if (! memoryPageSize) initPageSize();
    // align size up to the page size,
    // pvalloc(0) returns 1 page, see man libmpatrol
    size = size? ((size-1) | (memoryPageSize-1)) + 1 : memoryPageSize;

    return scalable_aligned_malloc(size, memoryPageSize);
}

int mallopt(int /*param*/, int /*value*/) __THROW
{
    return 1;
}

struct mallinfo mallinfo() __THROW
{
    struct mallinfo m;
    memset(&m, 0, sizeof(struct mallinfo));

    return m;
}

#if __ANDROID__
// Android doesn't have malloc_usable_size, provide it to be compatible
// with Linux, in addition overload dlmalloc_usable_size() that presented
// under Android.
size_t dlmalloc_usable_size(const void *ptr) __attribute__ ((alias ("malloc_usable_size")));
#else // __ANDROID__
// C11 function, supported starting GLIBC 2.16
void *aligned_alloc(size_t alignment, size_t size) __attribute__ ((alias ("memalign")));
// Those non-standard functions are exported by GLIBC, and might be used
// in conjunction with standard malloc/free, so we must ovberload them.
// Bionic doesn't have them. Not removing from the linker scripts,
// as absent entry points are ignored by the linker.
void *__libc_malloc(size_t size) __attribute__ ((alias ("malloc")));
void *__libc_calloc(size_t num, size_t size) __attribute__ ((alias ("calloc")));
void *__libc_memalign(size_t alignment, size_t size) __attribute__ ((alias ("memalign")));
void *__libc_pvalloc(size_t size) __attribute__ ((alias ("pvalloc")));
void *__libc_valloc(size_t size) __attribute__ ((alias ("valloc")));

// call original __libc_* to support naive replacement of free via __libc_free etc
void __libc_free(void *ptr)
{
    InitOrigPointers();
    __TBB_malloc_safer_free(ptr, (void (*)(void*))orig_libc_free);
}

void *__libc_realloc(void *ptr, size_t size)
{
    InitOrigPointers();
    return __TBB_malloc_safer_realloc(ptr, size, orig_libc_realloc);
}
#endif // !__ANDROID__

} /* extern "C" */

/*** replacements for global operators new and delete ***/

#include <new>

void * operator new(size_t sz) throw (std::bad_alloc) {
    void *res = scalable_malloc(sz);
#if TBB_USE_EXCEPTIONS
    if (NULL == res)
        throw std::bad_alloc();
#endif /* TBB_USE_EXCEPTIONS */
    return res;
}
void* operator new[](size_t sz) throw (std::bad_alloc) {
    void *res = scalable_malloc(sz);
#if TBB_USE_EXCEPTIONS
    if (NULL == res)
        throw std::bad_alloc();
#endif /* TBB_USE_EXCEPTIONS */
    return res;
}
void operator delete(void* ptr) throw() {
    InitOrigPointers();
    __TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void operator delete[](void* ptr) throw() {
    InitOrigPointers();
    __TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void* operator new(size_t sz, const std::nothrow_t&) throw() {
    return scalable_malloc(sz);
}
void* operator new[](std::size_t sz, const std::nothrow_t&) throw() {
    return scalable_malloc(sz);
}
void operator delete(void* ptr, const std::nothrow_t&) throw() {
    InitOrigPointers();
    __TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}
void operator delete[](void* ptr, const std::nothrow_t&) throw() {
    InitOrigPointers();
    __TBB_malloc_safer_free(ptr, (void (*)(void*))orig_free);
}

#endif /* MALLOC_UNIXLIKE_OVERLOAD_ENABLED */
#endif /* MALLOC_UNIXLIKE_OVERLOAD_ENABLED || MALLOC_ZONE_OVERLOAD_ENABLED */


#ifdef _WIN32
#include <windows.h>

#if !__TBB_WIN8UI_SUPPORT

#include <stdio.h>
#include "tbb_function_replacement.h"
#include "shared_utils.h"

void __TBB_malloc_safer_delete( void *ptr)
{
    __TBB_malloc_safer_free( ptr, NULL );
}

void* safer_aligned_malloc( size_t size, size_t alignment )
{
    // workaround for "is power of 2 pow N" bug that accepts zeros
    return scalable_aligned_malloc( size, alignment>sizeof(size_t*)?alignment:sizeof(size_t*) );
}

// we do not support _expand();
void* safer_expand( void *, size_t )
{
    return NULL;
}

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(CRTLIB)                                             \
void (*orig_free_##CRTLIB)(void*);                                                                   \
void __TBB_malloc_safer_free_##CRTLIB(void *ptr)                                                     \
{                                                                                                    \
    __TBB_malloc_safer_free( ptr, orig_free_##CRTLIB );                                              \
}                                                                                                    \
                                                                                                     \
void (*orig__aligned_free_##CRTLIB)(void*);                                                          \
void __TBB_malloc_safer__aligned_free_##CRTLIB(void *ptr)                                            \
{                                                                                                    \
    __TBB_malloc_safer_free( ptr, orig__aligned_free_##CRTLIB );                                     \
}                                                                                                    \
                                                                                                     \
size_t (*orig__msize_##CRTLIB)(void*);                                                               \
size_t __TBB_malloc_safer__msize_##CRTLIB(void *ptr)                                                 \
{                                                                                                    \
    return __TBB_malloc_safer_msize( ptr, orig__msize_##CRTLIB );                                    \
}                                                                                                    \
                                                                                                     \
size_t (*orig__aligned_msize_##CRTLIB)(void*, size_t, size_t);                                       \
size_t __TBB_malloc_safer__aligned_msize_##CRTLIB( void *ptr, size_t alignment, size_t offset)       \
{                                                                                                    \
    return __TBB_malloc_safer_aligned_msize( ptr, alignment, offset, orig__aligned_msize_##CRTLIB ); \
}                                                                                                    \
                                                                                                     \
void* __TBB_malloc_safer_realloc_##CRTLIB( void *ptr, size_t size )                                  \
{                                                                                                    \
    orig_ptrs func_ptrs = {orig_free_##CRTLIB, orig__msize_##CRTLIB};                                \
    return __TBB_malloc_safer_realloc( ptr, size, &func_ptrs );                                      \
}                                                                                                    \
                                                                                                     \
void* __TBB_malloc_safer__aligned_realloc_##CRTLIB( void *ptr, size_t size, size_t aligment )        \
{                                                                                                    \
    orig_aligned_ptrs func_ptrs = {orig__aligned_free_##CRTLIB, orig__aligned_msize_##CRTLIB};       \
    return __TBB_malloc_safer_aligned_realloc( ptr, size, aligment, &func_ptrs );                    \
}

// Only for ucrtbase: substitution for _o_free
void (*orig__o_free)(void*);
void __TBB_malloc__o_free(void *ptr)
{
    __TBB_malloc_safer_free( ptr, orig__o_free );
}
// Only for ucrtbase: substitution for _free_base
void(*orig__free_base)(void*);
void __TBB_malloc__free_base(void *ptr)
{
    __TBB_malloc_safer_free(ptr, orig__free_base);
}

// Size limit is MAX_PATTERN_SIZE (28) byte codes / 56 symbols per line.
// * can be used to match any digit in byte codes.
// # followed by several * indicate a relative address that needs to be corrected.
// Purpose of the pattern is to mark an instruction bound; it should consist of several
// full instructions plus one extra byte code. It's not required for the patterns
// to be unique (i.e., it's OK to have same pattern for unrelated functions).
// TODO: use hot patch prologues if exist
const char* known_bytecodes[] = {
#if _WIN64
//  "========================================================" - 56 symbols
    "4883EC284885C974",       // release free()
    "4883EC284885C975",       // release _msize()
    "4885C974375348",         // release free() 8.0.50727.42, 10.0
    "E907000000CCCC",         // release _aligned_msize(), _aligned_free() ucrtbase.dll
    "C7442410000000008B",     // release free() ucrtbase.dll 10.0.14393.33
    "E90B000000CCCC",         // release _msize() ucrtbase.dll 10.0.14393.33
    "48895C24085748",         // release _aligned_msize() ucrtbase.dll 10.0.14393.33
    "48894C24084883EC28BA",   // debug prologue
    "4C894424184889542410",   // debug _aligned_msize() 10.0
    "48894C24084883EC2848",   // debug _aligned_free 10.0
    "488BD1488D0D#*******E9", // _o_free(), ucrtbase.dll
 #if __TBB_OVERLOAD_OLD_MSVCR
    "48895C2408574883EC3049", // release _aligned_msize 9.0
    "4883EC384885C975",       // release _msize() 9.0
    "4C8BC1488B0DA6E4040033", // an old win64 SDK
 #endif
#else // _WIN32
//  "========================================================" - 56 symbols
    "8BFF558BEC8B",           // multiple
    "8BFF558BEC83",           // release free() & _msize() 10.0.40219.325, _msize() ucrtbase.dll
    "8BFF558BECFF",           // release _aligned_msize ucrtbase.dll
    "8BFF558BEC51",           // release free() & _msize() ucrtbase.dll 10.0.14393.33
    "558BEC8B450885C074",     // release _aligned_free 11.0
    "558BEC837D08000F",       // release _msize() 11.0.51106.1
    "558BEC837D08007419FF",   // release free() 11.0.50727.1
    "558BEC8B450885C075",     // release _aligned_msize() 11.0.50727.1
    "558BEC6A018B",           // debug free() & _msize() 11.0
    "558BEC8B451050",         // debug _aligned_msize() 11.0
    "558BEC8B450850",         // debug _aligned_free 11.0
    "8BFF558BEC6A",           // debug free() & _msize() 10.0.40219.325
 #if __TBB_OVERLOAD_OLD_MSVCR
    "6A1868********E8",       // release free() 8.0.50727.4053, 9.0
    "6A1C68********E8",       // release _msize() 8.0.50727.4053, 9.0
 #endif
#endif // _WIN64/_WIN32
    NULL
    };

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,function_name,dbgsuffix) \
    ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
      (FUNCPTR)__TBB_malloc_safer_##function_name##_##CRT_VER##dbgsuffix, \
      known_bytecodes, (FUNCPTR*)&orig_##function_name##_##CRT_VER##dbgsuffix );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,function_name,dbgsuffix) \
    ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
      (FUNCPTR)__TBB_malloc_safer_##function_name##_##CRT_VER##dbgsuffix, 0, NULL );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_REDIRECT(CRT_VER,function_name,dest_func,dbgsuffix) \
    ReplaceFunctionWithStore( #CRT_VER #dbgsuffix ".dll", #function_name, \
      (FUNCPTR)__TBB_malloc_safer_##dest_func##_##CRT_VER##dbgsuffix, 0, NULL );

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,dbgsuffix)                             \
    if (BytecodesAreKnown(#CRT_VER #dbgsuffix ".dll")) {                                          \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,free,dbgsuffix)                         \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_msize,dbgsuffix)                       \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,realloc,dbgsuffix)          \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_aligned_free,dbgsuffix)                \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY(CRT_VER,_aligned_msize,dbgsuffix)               \
      __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_ENTRY_NO_FALLBACK(CRT_VER,_aligned_realloc,dbgsuffix) \
    } else                                                                                        \
        SkipReplacement(#CRT_VER #dbgsuffix ".dll");

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(CRT_VER) __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,)
#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_DEBUG(CRT_VER) __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_IMPL(CRT_VER,d)

#define __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(CRT_VER)     \
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(CRT_VER) \
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_DEBUG(CRT_VER)

#if __TBB_OVERLOAD_OLD_MSVCR
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr70d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr70);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr71d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr71);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr80d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr80);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr90d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr90);
#endif
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr100d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr100);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr110d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr110);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr120d);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(msvcr120);
__TBB_ORIG_ALLOCATOR_REPLACEMENT_WRAPPER(ucrtbase);


/*** replacements for global operators new and delete ***/

#include <new>

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable : 4290 )
#endif

void * operator_new(size_t sz) throw (std::bad_alloc) {
    void *res = scalable_malloc(sz);
    if (NULL == res) throw std::bad_alloc();
    return res;
}
void* operator_new_arr(size_t sz) throw (std::bad_alloc) {
    void *res = scalable_malloc(sz);
    if (NULL == res) throw std::bad_alloc();
    return res;
}
void operator_delete(void* ptr) throw() {
    __TBB_malloc_safer_delete(ptr);
}
#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

void operator_delete_arr(void* ptr) throw() {
    __TBB_malloc_safer_delete(ptr);
}
void* operator_new_t(size_t sz, const std::nothrow_t&) throw() {
    return scalable_malloc(sz);
}
void* operator_new_arr_t(std::size_t sz, const std::nothrow_t&) throw() {
    return scalable_malloc(sz);
}
void operator_delete_t(void* ptr, const std::nothrow_t&) throw() {
    __TBB_malloc_safer_delete(ptr);
}
void operator_delete_arr_t(void* ptr, const std::nothrow_t&) throw() {
    __TBB_malloc_safer_delete(ptr);
}

struct Module {
    const char *name;
    bool        doFuncReplacement; // do replacement in the DLL
};

Module modules_to_replace[] = {
    {"msvcr100d.dll", true},
    {"msvcr100.dll", true},
    {"msvcr110d.dll", true},
    {"msvcr110.dll", true},
    {"msvcr120d.dll", true},
    {"msvcr120.dll", true},
    {"ucrtbase.dll", true},
//    "ucrtbased.dll" is not supported because of problems with _dbg functions
#if __TBB_OVERLOAD_OLD_MSVCR
    {"msvcr90d.dll", true},
    {"msvcr90.dll", true},
    {"msvcr80d.dll", true},
    {"msvcr80.dll", true},
    {"msvcr70d.dll", true},
    {"msvcr70.dll", true},
    {"msvcr71d.dll", true},
    {"msvcr71.dll", true},
#endif
#if __TBB_TODO
    // TODO: Try enabling replacement for non-versioned system binaries below
    {"msvcrtd.dll", true},
    {"msvcrt.dll", true},
#endif
    };

/*
We need to replace following functions:
malloc
calloc
_aligned_malloc
_expand (by dummy implementation)
??2@YAPAXI@Z      operator new                         (ia32)
??_U@YAPAXI@Z     void * operator new[] (size_t size)  (ia32)
??3@YAXPAX@Z      operator delete                      (ia32)
??_V@YAXPAX@Z     operator delete[]                    (ia32)
??2@YAPEAX_K@Z    void * operator new(unsigned __int64)   (intel64)
??_V@YAXPEAX@Z    void * operator new[](unsigned __int64) (intel64)
??3@YAXPEAX@Z     operator delete                         (intel64)
??_V@YAXPEAX@Z    operator delete[]                       (intel64)
??2@YAPAXIABUnothrow_t@std@@@Z      void * operator new (size_t sz, const std::nothrow_t&) throw()  (optional)
??_U@YAPAXIABUnothrow_t@std@@@Z     void * operator new[] (size_t sz, const std::nothrow_t&) throw() (optional)

and these functions have runtime-specific replacement:
realloc
free
_msize
_aligned_realloc
_aligned_free
_aligned_msize
*/

typedef struct FRData_t {
    //char *_module;
    const char *_func;
    FUNCPTR _fptr;
    FRR_ON_ERROR _on_error;
} FRDATA;

FRDATA c_routines_to_replace[] = {
    { "malloc",  (FUNCPTR)scalable_malloc, FRR_FAIL },
    { "calloc",  (FUNCPTR)scalable_calloc, FRR_FAIL },
    { "_aligned_malloc",  (FUNCPTR)safer_aligned_malloc, FRR_FAIL },
    { "_expand",  (FUNCPTR)safer_expand, FRR_IGNORE },
};

FRDATA cxx_routines_to_replace[] = {
#if _WIN64
    { "??2@YAPEAX_K@Z", (FUNCPTR)operator_new, FRR_FAIL },
    { "??_U@YAPEAX_K@Z", (FUNCPTR)operator_new_arr, FRR_FAIL },
    { "??3@YAXPEAX@Z", (FUNCPTR)operator_delete, FRR_FAIL },
    { "??_V@YAXPEAX@Z", (FUNCPTR)operator_delete_arr, FRR_FAIL },
#else
    { "??2@YAPAXI@Z", (FUNCPTR)operator_new, FRR_FAIL },
    { "??_U@YAPAXI@Z", (FUNCPTR)operator_new_arr, FRR_FAIL },
    { "??3@YAXPAX@Z", (FUNCPTR)operator_delete, FRR_FAIL },
    { "??_V@YAXPAX@Z", (FUNCPTR)operator_delete_arr, FRR_FAIL },
#endif
    { "??2@YAPAXIABUnothrow_t@std@@@Z", (FUNCPTR)operator_new_t, FRR_IGNORE },
    { "??_U@YAPAXIABUnothrow_t@std@@@Z", (FUNCPTR)operator_new_arr_t, FRR_IGNORE }
};

#ifndef UNICODE
typedef char unicode_char_t;
#define WCHAR_SPEC "%s"
#else
typedef wchar_t unicode_char_t;
#define WCHAR_SPEC "%ls"
#endif

// Check that we recognize bytecodes that should be replaced by trampolines.
// If some functions have unknown prologue patterns, replacement should not be done.
bool BytecodesAreKnown(const unicode_char_t *dllName)
{
    const char *funcName[] = {"free", "_msize", "_aligned_free", "_aligned_msize", 0};
    HMODULE module = GetModuleHandle(dllName);

    if (!module)
        return false;
    for (int i=0; funcName[i]; i++)
        if (! IsPrologueKnown(module, funcName[i], known_bytecodes)) {
            fprintf(stderr, "TBBmalloc: skip allocation functions replacement in " WCHAR_SPEC
                    ": unknown prologue for function " WCHAR_SPEC "\n", dllName, funcName[i]);
            return false;
        }
    return true;
}

void SkipReplacement(const unicode_char_t *dllName)
{
#ifndef UNICODE
    const char *dllStr = dllName;
#else
    const size_t sz = 128; // all DLL name must fit

    char buffer[sz];
    size_t real_sz;
    char *dllStr = buffer;

    errno_t ret = wcstombs_s(&real_sz, dllStr, sz, dllName, sz-1);
    __TBB_ASSERT(!ret, "Dll name conversion failed")
#endif

    for (size_t i=0; i<arrayLength(modules_to_replace); i++)
        if (!strcmp(modules_to_replace[i].name, dllStr)) {
            modules_to_replace[i].doFuncReplacement = false;
            break;
        }
}

void ReplaceFunctionWithStore( const unicode_char_t *dllName, const char *funcName, FUNCPTR newFunc, const char ** opcodes, FUNCPTR* origFunc,  FRR_ON_ERROR on_error = FRR_FAIL )
{
    FRR_TYPE res = ReplaceFunction( dllName, funcName, newFunc, opcodes, origFunc );

    if (res == FRR_OK || res == FRR_NODLL || (res == FRR_NOFUNC && on_error == FRR_IGNORE))
        return;

    fprintf(stderr, "Failed to %s function %s in module %s\n",
            res==FRR_NOFUNC? "find" : "replace", funcName, dllName);
    exit(1);
}

void doMallocReplacement()
{
    // Replace functions and keep backup of original code (separate for each runtime)
#if __TBB_OVERLOAD_OLD_MSVCR
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr70)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr71)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr80)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr90)
#endif
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr100)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr110)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL(msvcr120)
    __TBB_ORIG_ALLOCATOR_REPLACEMENT_CALL_RELEASE(ucrtbase)

    // Replace functions without storing original code
    for (size_t j = 0; j < arrayLength(modules_to_replace); j++) {
        if (!modules_to_replace[j].doFuncReplacement)
            continue;
        for (size_t i = 0; i < arrayLength(c_routines_to_replace); i++)
        {
            ReplaceFunctionWithStore( modules_to_replace[j].name, c_routines_to_replace[i]._func, c_routines_to_replace[i]._fptr, NULL, NULL,  c_routines_to_replace[i]._on_error );
        }
        if ( strcmp(modules_to_replace[j].name, "ucrtbase.dll") == 0 ) {
            // If _o_free function is present and patchable, redirect it to tbbmalloc as well
            // This prevents issues with other _o_* functions which might allocate memory with malloc
            if ( IsPrologueKnown(GetModuleHandle("ucrtbase.dll"), "_o_free", known_bytecodes) ) {
                ReplaceFunctionWithStore( "ucrtbase.dll", "_o_free", (FUNCPTR)__TBB_malloc__o_free, known_bytecodes, (FUNCPTR*)&orig__o_free,  FRR_FAIL );
            }
            // Similarly for _free_base
            if (IsPrologueKnown(GetModuleHandle("ucrtbase.dll"), "_free_base", known_bytecodes)) {
                ReplaceFunctionWithStore("ucrtbase.dll", "_free_base", (FUNCPTR)__TBB_malloc__free_base, known_bytecodes, (FUNCPTR*)&orig__free_base, FRR_FAIL);
            }
            // ucrtbase.dll does not export operator new/delete, so skip the rest of the loop.
            continue;
        }

        for (size_t i = 0; i < arrayLength(cxx_routines_to_replace); i++)
        {
#if !_WIN64
            // in Microsoft* Visual Studio* 2012 and 2013 32-bit operator delete consists of 2 bytes only: short jump to free(ptr);
            // replacement should be skipped for this particular case.
            if ( ((strcmp(modules_to_replace[j].name, "msvcr110.dll") == 0) || (strcmp(modules_to_replace[j].name, "msvcr120.dll") == 0)) && (strcmp(cxx_routines_to_replace[i]._func, "??3@YAXPAX@Z") == 0) ) continue;
            // in Microsoft* Visual Studio* 2013 32-bit operator delete[] consists of 2 bytes only: short jump to free(ptr);
            // replacement should be skipped for this particular case.
            if ( (strcmp(modules_to_replace[j].name, "msvcr120.dll") == 0) && (strcmp(cxx_routines_to_replace[i]._func, "??_V@YAXPAX@Z") == 0) ) continue;
#endif
            ReplaceFunctionWithStore( modules_to_replace[j].name, cxx_routines_to_replace[i]._func, cxx_routines_to_replace[i]._fptr, NULL, NULL,  cxx_routines_to_replace[i]._on_error );
        }
    }
}

#endif // !__TBB_WIN8UI_SUPPORT

extern "C" BOOL WINAPI DllMain( HINSTANCE hInst, DWORD callReason, LPVOID reserved )
{

    if ( callReason==DLL_PROCESS_ATTACH && reserved && hInst ) {
#if !__TBB_WIN8UI_SUPPORT
#if TBBMALLOC_USE_TBB_FOR_ALLOCATOR_ENV_CONTROLLED
        char pinEnvVariable[50];
        if( GetEnvironmentVariable("TBBMALLOC_USE_TBB_FOR_ALLOCATOR", pinEnvVariable, 50))
        {
            doMallocReplacement();
        }
#else
        doMallocReplacement();
#endif
#endif // !__TBB_WIN8UI_SUPPORT
    }

    return TRUE;
}

// Just to make the linker happy and link the DLL to the application
extern "C" __declspec(dllexport) void __TBB_malloc_proxy()
{

}

#endif //_WIN32
