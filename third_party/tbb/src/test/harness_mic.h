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

#ifndef tbb_test_harness_mic_H
#define tbb_test_harness_mic_H

#if ! __TBB_DEFINE_MIC
    #error test/harness_mic.h should be included only when building for Intel(R) Many Integrated Core Architecture
#endif

// test for unifed sources. See makefiles
#undef HARNESS_INCOMPLETE_SOURCES

#include <stdlib.h>
#include <stdio.h>

#define TBB_TEST_LOW_WORKLOAD 1

#define REPORT_FATAL_ERROR  REPORT
#define HARNESS_EXPORT

#if __TBB_MIC_NATIVE
    #define HARNESS_EXIT_ON_ASSERT 1
    #define __TBB_PLACEMENT_NEW_EXCEPTION_SAFETY_BROKEN 1
#else
    #define HARNESS_TERMINATE_ON_ASSERT 1
#endif

#endif /* tbb_test_harness_mic_H */
