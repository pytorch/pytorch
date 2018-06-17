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

#include "tbb/tbb_thread.h"
#define THREAD tbb::tbb_thread
#define THIS_THREAD tbb::this_tbb_thread
#define THIS_THREAD_SLEEP THIS_THREAD::sleep
#include "test_thread.h"
#include "harness.h"

/* we want to test tbb::tbb_thread */
int TestMain () {
    CheckSignatures();
    RunTests();
    return Harness::Done;
}
