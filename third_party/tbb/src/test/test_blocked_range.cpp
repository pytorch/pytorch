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

#include "tbb/blocked_range.h"
#include "harness_assert.h"

// First test as much as we can without including other headers.
// Doing so should catch problems arising from failing to include headers.

class AbstractValueType {
    AbstractValueType() {}
    int value;
public:
    friend AbstractValueType MakeAbstractValueType( int i );
    friend int GetValueOf( const AbstractValueType& v ) {return v.value;}
};

AbstractValueType MakeAbstractValueType( int i ) {
    AbstractValueType x;
    x.value = i;
    return x;
}

std::size_t operator-( const AbstractValueType& u, const AbstractValueType& v ) {
    return GetValueOf(u) - GetValueOf(v);
}

bool operator<( const AbstractValueType& u, const AbstractValueType& v ) {
    return GetValueOf(u) < GetValueOf(v);
}

AbstractValueType operator+( const AbstractValueType& u, std::size_t offset ) {
    return MakeAbstractValueType(GetValueOf(u) + int(offset));
}

static void SerialTest() {
    for( int x=-10; x<10; ++x )
        for( int y=-10; y<10; ++y ) {
            AbstractValueType i = MakeAbstractValueType(x);
            AbstractValueType j = MakeAbstractValueType(y);
            for( std::size_t k=1; k<10; ++k ) {
                typedef tbb::blocked_range<AbstractValueType> range_type;
                range_type r( i, j, k );
                AssertSameType( r.empty(), true );
                AssertSameType( range_type::size_type(), std::size_t() );
                AssertSameType( static_cast<range_type::const_iterator*>(0), static_cast<AbstractValueType*>(0) );
                AssertSameType( r.begin(), MakeAbstractValueType(0) );
                AssertSameType( r.end(), MakeAbstractValueType(0) );
                ASSERT( r.empty()==(y<=x), NULL );
                ASSERT( r.grainsize()==k, NULL );
                if( x<=y ) {
                    AssertSameType( r.is_divisible(), true );
                    ASSERT( r.is_divisible()==(std::size_t(y-x)>k), NULL );
                    ASSERT( r.size()==std::size_t(y-x), NULL );
                    if( r.is_divisible() ) {
                        tbb::blocked_range<AbstractValueType> r2(r,tbb::split());
                        ASSERT( GetValueOf(r.begin())==x, NULL );
                        ASSERT( GetValueOf(r.end())==GetValueOf(r2.begin()), NULL );
                        ASSERT( GetValueOf(r2.end())==y, NULL );
                        ASSERT( r.grainsize()==k, NULL );
                        ASSERT( r2.grainsize()==k, NULL );
                    }
                }
            }
        }
}

#include "tbb/parallel_for.h"
#include "harness.h"

const int N = 1<<22;

unsigned char Array[N];

struct Striker {
    // Note: we use <int> here instead of <long> in order to test for Quad 407676
    void operator()( const tbb::blocked_range<int>& r ) const {
        for( tbb::blocked_range<int>::const_iterator i=r.begin(); i!=r.end(); ++i )
            ++Array[i];
    }
};

void ParallelTest() {
    for( int i=0; i<N; i=i<3 ? i+1 : i*3 ) {
        const tbb::blocked_range<int> r( 0, i, 10 );
        tbb::parallel_for( r, Striker() );
        for( int k=0; k<N; ++k ) {
            ASSERT( Array[k]==(k<i), NULL );
            Array[k] = 0;
        }
    }
}

#if __TBB_RANGE_BASED_FOR_PRESENT
#include "test_range_based_for.h"
#include <functional>
void TestRangeBasedFor() {
    using namespace range_based_for_support_tests;
    REMARK("testing range based for loop compatibility \n");

    size_t int_array[100] = {0};
    const size_t sequence_length = Harness::array_length(int_array);

    for (size_t i = 0; i < sequence_length; ++i) {
        int_array[i] = i + 1;
    }

    const tbb::blocked_range<size_t*> r(int_array, Harness::end(int_array), 1);

    ASSERT(range_based_for_accumulate<size_t>(r, std::plus<size_t>(), size_t(0)) == gauss_summ_of_int_sequence(sequence_length), "incorrect accumulated value generated via range based for ?");
}
#endif //if __TBB_RANGE_BASED_FOR_PRESENT

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES

void TestProportionalSplitOverflow()
{
    REMARK("Testing overflow during proportional split - ");
    using tbb::blocked_range;
    using tbb::proportional_split;

    blocked_range<size_t> r1(0, size_t(-1) / 2);
    size_t size = r1.size();
    size_t begin = r1.begin();
    size_t end = r1.end();

    proportional_split p(1, 3);
    blocked_range<size_t> r2(r1, p);

    // overflow-free computation
    size_t parts = p.left() + p.right();
    size_t int_part = size / parts;
    size_t fraction = size - int_part * parts; // fraction < parts
    size_t right_idx = int_part * p.right() + fraction * p.right() / parts + 1;
    size_t newRangeBegin = end - right_idx;

    // Division in 'right_idx' very likely is inexact also.
    size_t tolerance = 1;
    size_t diff = (r2.begin() < newRangeBegin) ? (newRangeBegin - r2.begin()) : (r2.begin() - newRangeBegin);
    bool is_split_correct = diff <= tolerance;
    bool test_passed = (r1.begin() == begin && r1.end() == r2.begin() && is_split_correct &&
                        r2.end() == end);
    if (!test_passed) {
        REPORT("Incorrect split of blocked range[%lu, %lu) into r1[%lu, %lu) and r2[%lu, %lu), "
               "must be r1[%lu, %lu) and r2[%lu, %lu)\n", begin, end, r1.begin(), r1.end(), r2.begin(), r2.end(), begin, newRangeBegin, newRangeBegin, end);
        ASSERT(test_passed, NULL);
    }
    REMARK("OK\n");
}
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */
//------------------------------------------------------------------------
// Test driver
#include "tbb/task_scheduler_init.h"

int TestMain () {
    SerialTest();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        ParallelTest();
    }

    #if __TBB_RANGE_BASED_FOR_PRESENT
        TestRangeBasedFor();
    #endif //if __TBB_RANGE_BASED_FOR_PRESENT

    #if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
        TestProportionalSplitOverflow();
    #endif

    return Harness::Done;
}
