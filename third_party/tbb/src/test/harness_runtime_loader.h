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

#ifndef harness_runtime_loader_H
#define harness_runtime_loader_H

#if HARNESS_USE_RUNTIME_LOADER
    #if TEST_USES_TBB
        #define TBB_PREVIEW_RUNTIME_LOADER 1
        #include "tbb/runtime_loader.h"
        static char const * _path[] = { ".", NULL };
        // declaration must be placed before 1st TBB call
        static tbb::runtime_loader _runtime_loader( _path );
    #else // TEST_USES_TBB
        // if TBB library is not used, no need to test Runtime Loader
        #define HARNESS_SKIP_TEST 1
    #endif // TEST_USES_TBB
#endif // HARNESS_USE_RUNTIME_LOADER

#endif /* harness_runtime_loader_H */
