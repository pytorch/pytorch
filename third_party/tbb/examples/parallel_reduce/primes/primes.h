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

#ifndef PRIMES_H_
#define PRIMES_H_

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (push,target(mic))
#endif // __TBB_MIC_OFFLOAD

#include "tbb/task_scheduler_init.h"
#include <cstddef>
typedef std::size_t NumberType;

//! Count number of primes between 0 and n
/** This is the serial version. */
NumberType SerialCountPrimes( NumberType n);

//! Count number of primes between 0 and n
/** This is the parallel version. */
NumberType ParallelCountPrimes( NumberType n, int numberOfThreads= tbb::task_scheduler_init::automatic, NumberType grainSize = 1000);

#if __TBB_MIC_OFFLOAD
#pragma offload_attribute (pop)
#endif // __TBB_MIC_OFFLOAD

#endif /* PRIMES_H_ */
