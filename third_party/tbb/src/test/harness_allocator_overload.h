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

#ifndef tbb_test_harness_allocator_overload_H
#define tbb_test_harness_allocator_overload_H

#include "../tbbmalloc/proxy.h" // for MALLOC_UNIXLIKE_OVERLOAD_ENABLED, MALLOC_ZONE_OVERLOAD_ENABLED
#include "tbb/tbb_config.h" // for __TBB_WIN8UI_SUPPORT

// Skip configurations with unsupported system malloc overload:
// skip unsupported MSVCs, WIN8UI and MINGW (it doesn't define _MSC_VER),
// no support for MSVC 2015 and greater in debug for now,
// don't use defined(_MSC_VER), because result of using defined() in macro expansion is undefined
#define MALLOC_WINDOWS_OVERLOAD_ENABLED ((_WIN32||_WIN64) && !__TBB_WIN8UI_SUPPORT && _MSC_VER >= 1500 && !(_MSC_VER >= 1900 && _DEBUG))

// Skip configurations with unsupported system malloc overload:
// * overload via linking with -lmalloc_proxy is broken in offload,
// as the library is loaded too late in that mode,
// * LD_PRELOAD mechanism is broken in offload
#define HARNESS_SKIP_TEST ((!MALLOC_WINDOWS_OVERLOAD_ENABLED && !MALLOC_UNIXLIKE_OVERLOAD_ENABLED && !MALLOC_ZONE_OVERLOAD_ENABLED) || __TBB_MIC_OFFLOAD)

#endif // tbb_test_harness_allocator_overload_H
