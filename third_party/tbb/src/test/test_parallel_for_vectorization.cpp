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

// The test checks if the vectorization happens when PPL-style parallel_for is
// used. The test implements two ideas:
// 1. "pragma always assert" issues a compiler-time error if the vectorization
// cannot be produced;
// 2. "#pragma ivdep" has a peculiarity which also can be used for detection of
// successful vectorization. See the comment below.

// For now, only Intel(R) C++ Compiler 14.0 and later is supported. Also, no
// sense to run the test in debug mode.
#define HARNESS_SKIP_TEST ( __INTEL_COMPILER < 1400  || TBB_USE_DEBUG )

// __TBB_ASSERT_ON_VECTORIZATION_FAILURE enables "pragma always assert" for
// Intel(R) C++ Compiler.
#define __TBB_ASSERT_ON_VECTORIZATION_FAILURE ( !HARNESS_SKIP_TEST )
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

#include "harness.h"
#include "harness_assert.h"

#include <algorithm>

class Body : NoAssign {
    int *out_, *in_;
public:
    Body( int* out, int *in ) : out_(out), in_(in) {}
    void operator() ( int i ) const {
        out_[i] = in_[i] + 1;
    }
};

int TestMain () {
    // Should be big enough that the partitioner generated at least a one range
    // with a size greater than 1. See the comment below.
    const int N = 10000;
    tbb::task_scheduler_init init(1);
    int array1[N];
    std::fill( array1, array1+N, 0 );
    // Use the same array (with a shift) for both input and output
    tbb::parallel_for( 0, N-1, Body(array1+1, array1) );

    int array2[N];
    std::fill( array2, array2+N, 0 );
    Body b(array2+1, array2);
    for ( int i=0; i<N-1; ++i )
        b(i);

    // The ppl-style parallel_for implementation has pragma ivdep before the
    // range loop. This pragma suppresses the dependency of overlapping arrays
    // in "Body". Thus the vectorizer should generate code that produces incorrect
    // results.
    ASSERT( !std::equal( array1, array1+N, array2 ), "The loop was not vectorized." );

    return  Harness::Done;
}
