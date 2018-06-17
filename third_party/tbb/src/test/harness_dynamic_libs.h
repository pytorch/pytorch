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

// Include this header file before harness.h for HARNESS_SKIP_TEST to take effect
#if !__TBB_DYNAMIC_LOAD_ENABLED
#define HARNESS_SKIP_TEST 1
#else

#if _WIN32 || _WIN64
#include "tbb/machine/windows_api.h"
#else
#include <dlfcn.h>
#endif
#include "harness_assert.h"

namespace Harness {

#if TBB_USE_DEBUG
#define SUFFIX1 "_debug"
#define SUFFIX2
#else
#define SUFFIX1
#define SUFFIX2 "_debug"
#endif /* TBB_USE_DEBUG */

#if _WIN32||_WIN64
#define PREFIX
#define EXT ".dll"
#else
#define PREFIX "lib"
#if __APPLE__
#define EXT ".dylib"
// Android SDK build system does not support .so file name versioning
#elif __FreeBSD__ || __NetBSD__ || __sun || _AIX || __ANDROID__
#define EXT ".so"
#elif __linux__  // Order of these elif's matters!
#define EXT __TBB_STRING(.so.TBB_COMPATIBLE_INTERFACE_VERSION)
#else
#error Unknown OS
#endif
#endif

// Form the names of the TBB memory allocator binaries.
#define MALLOCLIB_NAME1 PREFIX "tbbmalloc" SUFFIX1 EXT
#define MALLOCLIB_NAME2 PREFIX "tbbmalloc" SUFFIX2 EXT

#if _WIN32 || _WIN64
typedef  HMODULE LIBRARY_HANDLE;
#else
typedef void *LIBRARY_HANDLE;
#endif

#if _WIN32 || _WIN64
#define TEST_LIBRARY_NAME(base) base".dll"
#elif __APPLE__
#define TEST_LIBRARY_NAME(base) base".dylib"
#else
#define TEST_LIBRARY_NAME(base) base".so"
#endif

LIBRARY_HANDLE OpenLibrary(const char *name)
{
#if _WIN32 || _WIN64
#if __TBB_WIN8UI_SUPPORT
    TCHAR wlibrary[MAX_PATH];
    if ( MultiByteToWideChar(CP_UTF8, 0, name, -1, wlibrary, MAX_PATH) == 0 ) return false;
    return :: LoadPackagedLibrary( wlibrary, 0 );
#else
    return ::LoadLibrary(name);
#endif
#else
    return dlopen(name, RTLD_NOW|RTLD_GLOBAL);
#endif
}

void CloseLibrary(LIBRARY_HANDLE lib)
{
#if _WIN32 || _WIN64
    BOOL ret = FreeLibrary(lib);
    ASSERT(ret, "FreeLibrary must be successful");
#else
    int ret = dlclose(lib);
    ASSERT(ret == 0, "dlclose must be successful");
#endif
}

typedef void (*FunctionAddress)();

FunctionAddress GetAddress(Harness::LIBRARY_HANDLE lib, const char *name)
{
    union { FunctionAddress func; void *symb; } converter;
#if _WIN32 || _WIN64
    converter.symb = (void*)GetProcAddress(lib, name);
#else
    converter.symb = (void*)dlsym(lib, name);
#endif
    ASSERT(converter.func, "Can't find required symbol in dynamic library");
    return converter.func;
}

}  // namespace Harness

#endif // __TBB_DYNAMIC_LOAD_ENABLED
