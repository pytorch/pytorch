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

// Test whether cache_aligned_allocator works with some of the host's STL containers.

#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_allocator.h"

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "test_allocator_STL.h"

int TestMain () {
    TestAllocatorWithSTL<tbb::cache_aligned_allocator<void> >();
    TestAllocatorWithSTL<tbb::tbb_allocator<void> >();
    TestAllocatorWithSTL<tbb::zero_allocator<void> >();
    return Harness::Done;
}

