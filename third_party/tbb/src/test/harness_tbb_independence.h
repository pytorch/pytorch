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

#ifndef harness_tbb_independence_H
#define harness_tbb_independence_H

// The tests which include tbb/atomic.h gain the dependency on the __TBB_ASSERT
// implementation even the test does not use anything from it. But almost all
// compilers optimize out unused inline function so they throw out the
// dependency. But to be pedantic with the standard the __TBB_ASSERT
// implementation should be provided. Moreover the offload compiler really
// requires it.
#include "../tbb/tbb_assert_impl.h"

#if __linux__  && __ia64__

#define __TBB_NO_IMPLICIT_LINKAGE 1
#include "tbb/tbb_machine.h"

#include <pthread.h>

// Can't use Intel compiler intrinsic due to internal error reported by 10.1 compiler
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

int32_t __TBB_machine_fetchadd4__TBB_full_fence (volatile void *ptr, int32_t value)
{
    pthread_mutex_lock(&counter_mutex);
    int32_t result = *(int32_t*)ptr;
    *(int32_t*)ptr = result + value;
    pthread_mutex_unlock(&counter_mutex);
    return result;
}

int64_t __TBB_machine_fetchadd8__TBB_full_fence (volatile void *ptr, int64_t value)
{
    pthread_mutex_lock(&counter_mutex);
    int32_t result = *(int32_t*)ptr;
    *(int32_t*)ptr = result + value;
    pthread_mutex_unlock(&counter_mutex);
    return result;
}

void __TBB_machine_pause(int32_t /*delay*/) {  __TBB_Yield(); }

pthread_mutex_t cas_mutex = PTHREAD_MUTEX_INITIALIZER;

extern "C" int64_t __TBB_machine_cmpswp8__TBB_full_fence(volatile void *ptr, int64_t value, int64_t comparand)
{
    pthread_mutex_lock(&cas_mutex);
    int64_t result = *(int64_t*)ptr;
    if (result == comparand)
        *(int64_t*)ptr = value;
    pthread_mutex_unlock(&cas_mutex);
    return result;
}

pthread_mutex_t fetchstore_mutex = PTHREAD_MUTEX_INITIALIZER;

int64_t __TBB_machine_fetchstore8__TBB_full_fence (volatile void *ptr, int64_t value)
{
    pthread_mutex_lock(&fetchstore_mutex);
    int64_t result = *(int64_t*)ptr;
    *(int64_t*)ptr = value;
    pthread_mutex_unlock(&fetchstore_mutex);
    return result;
}

#endif /* __linux__  && __ia64 */

#endif // harness_tbb_independence_H
