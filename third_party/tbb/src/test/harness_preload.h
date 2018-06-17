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

// This file is intended for preloading (via compiler options such as -include) into every test.
// Alas, not all compilers have such options, so the file is "optional".

// Only add here things that are necessary for *every* test!
// In particular, avoid including other headers.
// Since this file can be omitted, checking compiler-specific conditions is strongly recommended.

#ifndef harness_preload_H
#define harness_preload_H

#if __GNUC__>=5 && !__INTEL_COMPILER && !__clang__ && __GXX_EXPERIMENTAL_CXX0X__
// GCC 5 has added -Wsuggest-override, but unfortunately enables it even in pre-C++11 mode.
// We only want to use it for C++11 though.
#pragma GCC diagnostic warning "-Wsuggest-override"
#define __TBB_TEST_USE_WSUGGEST_OVERRIDE 1
#endif
// TODO: consider adding a similar option for clang

#if __TBB_TEST_NO_EXCEPTIONS
// This code breaks our own recommendations above, and it's deliberate:
// it includes another file, but that file should only have macros and pragmas;
// it does not check for compiler, as that is checked in the included file.
// The file also defines TBB_USE_EXCEPTIONS=0, which is set for all tests via makefiles anyway.
#include "tbb/tbb_disable_exceptions.h"
#endif

#endif /* harness_preload_H */
