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


#include "tbb/parallel_reduce.h"
#include "tbb/atomic.h"
#include "harness_assert.h"

using namespace std;

static tbb::atomic<long> ForkCount;
static tbb::atomic<long> FooBodyCount;

//! Class with public interface that is exactly minimal requirements for Range concept
class MinimalRange {
    size_t begin, end;
    friend class FooBody;
    explicit MinimalRange( size_t i ) : begin(0), end(i) {}
    friend void Flog( int nthread, bool inteference );
public:
    MinimalRange( MinimalRange& r, tbb::split ) : end(r.end) {
        begin = r.end = (r.begin+r.end)/2;
    }
    bool is_divisible() const {return end-begin>=2;}
    bool empty() const {return begin==end;}
};

//! Class with public interface that is exactly minimal requirements for Body of a parallel_reduce
class FooBody {
private:
    FooBody( const FooBody& );          // Deny access
    void operator=( const FooBody& );   // Deny access
    friend void Flog( int nthread, bool interference );
    //! Parent that created this body via split operation.  NULL if original body.
    FooBody* parent;
    //! Total number of index values processed by body and its children.
    size_t sum;
    //! Number of join operations done so far on this body and its children.
    long join_count;
    //! Range that has been processed so far by this body and its children.
    size_t begin, end;
    //! True if body has not yet been processed at least once by operator().
    bool is_new;
    //! 1 if body was created by split; 0 if original body.
    int forked;
    FooBody() {++FooBodyCount;}
public:
    ~FooBody() {
        forked = 0xDEADBEEF;
        sum=0xDEADBEEF;
        join_count=0xDEADBEEF;
        --FooBodyCount;
    }
    FooBody( FooBody& other, tbb::split ) {
        ++FooBodyCount;
        ++ForkCount;
        sum = 0;
        parent = &other;
        join_count = 0;
        is_new = true;
        forked = 1;
    }
    void join( FooBody& s ) {
        ASSERT( s.forked==1, NULL );
        ASSERT( this!=&s, NULL );
        ASSERT( this==s.parent, NULL );
        ASSERT( end==s.begin, NULL );
        end = s.end;
        sum += s.sum;
        join_count += s.join_count + 1;
        s.forked = 2;
    }
    void operator()( const MinimalRange& r ) {
        for( size_t k=r.begin; k<r.end; ++k )
            ++sum;
        if( is_new ) {
            is_new = false;
            begin = r.begin;
        } else
            ASSERT( end==r.begin, NULL );
        end = r.end;
    }
};

#include <cstdio>
#include "harness.h"
#include "tbb/tick_count.h"

void Flog( int nthread, bool interference=false ) {
    for (int mode = 0;  mode < 4; mode++) {
        tbb::tick_count T0 = tbb::tick_count::now();
        long join_count = 0;
        tbb::affinity_partitioner ap;
        for( size_t i=0; i<=1000; ++i ) {
            FooBody f;
            f.sum = 0;
            f.parent = NULL;
            f.join_count = 0;
            f.is_new = true;
            f.forked = 0;
            f.begin = ~size_t(0);
            f.end = ~size_t(0);
            ASSERT( FooBodyCount==1, NULL );
            switch (mode) {
                case 0:
                    tbb::parallel_reduce( MinimalRange(i), f );
                    break;
                case 1:
                    tbb::parallel_reduce( MinimalRange(i), f, tbb::simple_partitioner() );
                    break;
                case 2:
                    tbb::parallel_reduce( MinimalRange(i), f, tbb::auto_partitioner() );
                    break;
                case 3:
                    tbb::parallel_reduce( MinimalRange(i), f, ap );
                    break;
            }
            join_count += f.join_count;
            ASSERT( FooBodyCount==1, NULL );
            ASSERT( f.sum==i, NULL );
            ASSERT( f.begin==(i==0 ? ~size_t(0) : 0), NULL );
            ASSERT( f.end==(i==0 ? ~size_t(0) : i), NULL );
        }
        tbb::tick_count T1 = tbb::tick_count::now();
        REMARK("time=%g join_count=%ld ForkCount=%ld nthread=%d%s\n",
                   (T1-T0).seconds(),join_count,long(ForkCount), nthread, interference ? " with interference":"");
    }
}

#include "tbb/blocked_range.h"

#if _MSC_VER
    typedef tbb::internal::uint64_t ValueType;
#else
    typedef uint64_t ValueType;
#endif

struct Sum {
    template<typename T>
    T operator() ( const T& v1, const T& v2 ) const {
        return v1 + v2;
    }
};

struct Accumulator {
    ValueType operator() ( const tbb::blocked_range<ValueType*>& r, ValueType value ) const {
        for ( ValueType* pv = r.begin(); pv != r.end(); ++pv )
            value += *pv;
        return value;
    }
};

class ParallelSumTester: public NoAssign {
public:
    ParallelSumTester() : m_range(NULL, NULL) {
        m_array = new ValueType[unsigned(N)];
        for ( ValueType i = 0; i < N; ++i )
            m_array[i] = i + 1;
        m_range = tbb::blocked_range<ValueType*>( m_array, m_array + N );
    }
    ~ParallelSumTester() { delete[] m_array; }
    template<typename Partitioner>
    void CheckParallelReduce() {
        Partitioner partitioner;
        ValueType r1 = tbb::parallel_reduce( m_range, I, Accumulator(), Sum(), partitioner );
        ASSERT( r1 == R, NULL );
#if __TBB_CPP11_LAMBDAS_PRESENT
        ValueType r2 = tbb::parallel_reduce(
            m_range, I,
            [](const tbb::blocked_range<ValueType*>& r, ValueType value) -> ValueType {
                for ( const ValueType* pv = r.begin(); pv != r.end(); ++pv )
                    value += *pv;
                return value;
            },
            Sum(),
            partitioner
        );
        ASSERT( r2 == R, NULL );
#endif /* LAMBDAS */
    }
    void CheckParallelReduceDefault() {
        ValueType r1 = tbb::parallel_reduce( m_range, I, Accumulator(), Sum() );
        ASSERT( r1 == R, NULL );
#if __TBB_CPP11_LAMBDAS_PRESENT
        ValueType r2 = tbb::parallel_reduce(
            m_range, I,
            [](const tbb::blocked_range<ValueType*>& r, ValueType value) -> ValueType {
                for ( const ValueType* pv = r.begin(); pv != r.end(); ++pv )
                    value += *pv;
                return value;
            },
            Sum()
        );
        ASSERT( r2 == R, NULL );
#endif /* LAMBDAS */
    }
private:
    ValueType* m_array;
    tbb::blocked_range<ValueType*> m_range;
    static const ValueType I, N, R;
};

const ValueType ParallelSumTester::I = 0;
const ValueType ParallelSumTester::N = 1000000;
const ValueType ParallelSumTester::R = N * (N + 1) / 2;

void ParallelSum () {
    ParallelSumTester pst;
    pst.CheckParallelReduceDefault();
    pst.CheckParallelReduce<tbb::simple_partitioner>();
    pst.CheckParallelReduce<tbb::auto_partitioner>();
    pst.CheckParallelReduce<tbb::affinity_partitioner>();
    pst.CheckParallelReduce<tbb::static_partitioner>();
}

#include "harness_concurrency_tracker.h"

class RotOp {
public:
    typedef int Type;
    int operator() ( int x, int i ) const {
        return ( x<<1 ) ^ i;
    }
    int join( int x, int y ) const {
        return operator()( x, y );
    }
};

template <class Op>
struct ReduceBody {
    typedef typename Op::Type result_type;
    result_type my_value;

    ReduceBody() : my_value() {}
    ReduceBody( ReduceBody &, tbb::split ) : my_value() {}

    void operator() ( const tbb::blocked_range<int>& r ) {
        Harness::ConcurrencyTracker ct;
        for ( int i = r.begin(); i != r.end(); ++i ) {
            Op op;
            my_value = op(my_value, i);
        }
    }

    void join( const ReduceBody& y ) {
        Op op;
        my_value = op.join(my_value, y.my_value);
    }
};

//! Type-tag for automatic testing algorithm deduction
struct harness_default_partitioner {};

template<typename Body, typename Partitioner>
struct parallel_deterministic_reduce_invoker {
    template<typename Range>
    static typename Body::result_type run( const Range& range ) {
        Body body;
        tbb::parallel_deterministic_reduce(range, body, Partitioner());
        return body.my_value;
    }
};

template<typename Body>
struct parallel_deterministic_reduce_invoker<Body, harness_default_partitioner> {
    template<typename Range>
    static typename Body::result_type run( const Range& range ) {
        Body body;
        tbb::parallel_deterministic_reduce(range, body);
        return body.my_value;
    }
};

template<typename ResultType, typename Partitioner>
struct parallel_deterministic_reduce_lambda_invoker {
    template<typename Range, typename Func, typename Reduction>
    static ResultType run( const Range& range, Func f, Reduction r ) {
        return tbb::parallel_deterministic_reduce(range, ResultType(), f, r, Partitioner());
    }
};

template<typename ResultType>
struct parallel_deterministic_reduce_lambda_invoker<ResultType, harness_default_partitioner> {
    template<typename Range, typename Func, typename Reduction>
    static ResultType run(const Range& range, Func f, Reduction r) {
        return tbb::parallel_deterministic_reduce(range, ResultType(), f, r);
    }
};

//! Define overloads of parallel_deterministic_reduce that accept "undesired" types of partitioners
namespace unsupported {

    template<typename Range, typename Body>
    void parallel_deterministic_reduce(const Range&, Body&, const tbb::auto_partitioner&) { }

    template<typename Range, typename Body>
    void parallel_deterministic_reduce(const Range&, Body&, tbb::affinity_partitioner&) { }

    template<typename Range, typename Value, typename RealBody, typename Reduction>
    Value parallel_deterministic_reduce(const Range& , const Value& identity, const RealBody& , const Reduction& , const tbb::auto_partitioner&) {
        return identity;
    }

    template<typename Range, typename Value, typename RealBody, typename Reduction>
    Value parallel_deterministic_reduce(const Range& , const Value& identity, const RealBody& , const Reduction& , tbb::affinity_partitioner&) {
        return identity;
    }

}

struct Body {
    float value;
    Body() : value(0) {}
    Body(Body&, tbb::split) { value = 0; }
    void operator()(const tbb::blocked_range<int>&) {}
    void join(Body&) {}
};

//! Check that other types of partitioners are not supported (auto, affinity)
//! In the case of "unsupported" API unexpectedly sneaking into namespace tbb,
//! this test should result in a compilation error due to overload resolution ambiguity
static void TestUnsupportedPartitioners() {
    using namespace tbb;
    using namespace unsupported;
    Body body;
    parallel_deterministic_reduce(blocked_range<int>(0, 10), body, tbb::auto_partitioner());

    tbb::affinity_partitioner ap;
    parallel_deterministic_reduce(blocked_range<int>(0, 10), body, ap);

#if __TBB_CPP11_LAMBDAS_PRESENT
    parallel_deterministic_reduce(
        blocked_range<int>(0, 10),
        0,
        [](const blocked_range<int>&, int init)->int {
            return init;
        },
        [](int x, int y)->int {
            return x + y;
        },
        tbb::auto_partitioner()
    );
    parallel_deterministic_reduce(
        blocked_range<int>(0, 10),
        0,
        [](const blocked_range<int>&, int init)->int {
            return init;
        },
        [](int x, int y)->int {
            return x + y;
        },
        ap
    );
#endif /* LAMBDAS */
}

template <class Partitioner>
void TestDeterministicReductionFor() {
    const int N = 1000;
    const tbb::blocked_range<int> range(0, N);
    typedef ReduceBody<RotOp> BodyType;
    BodyType::result_type R1 =
        parallel_deterministic_reduce_invoker<BodyType, Partitioner>::run(range);
    for ( int i=0; i<100; ++i ) {
        BodyType::result_type R2 =
            parallel_deterministic_reduce_invoker<BodyType, Partitioner>::run(range);
        ASSERT( R1 == R2, "parallel_deterministic_reduce behaves differently from run to run" );
#if __TBB_CPP11_LAMBDAS_PRESENT
        typedef RotOp::Type Type;
        Type R3 = parallel_deterministic_reduce_lambda_invoker<Type, Partitioner>::run(
            range,
            [](const tbb::blocked_range<int>& br, Type value) -> Type {
                Harness::ConcurrencyTracker ct;
                for ( int ii = br.begin(); ii != br.end(); ++ii ) {
                    RotOp op;
                    value = op(value, ii);
                }
                return value;
            },
            [](const Type& v1, const Type& v2) -> Type {
                RotOp op;
                return op.join(v1,v2);
            }
        );
        ASSERT( R1 == R3, "lambda-based parallel_deterministic_reduce behaves differently from run to run" );
#endif /* LAMBDAS */
    }
}

void TestDeterministicReduction () {
    TestDeterministicReductionFor<tbb::simple_partitioner>();
    TestDeterministicReductionFor<tbb::static_partitioner>();
    TestDeterministicReductionFor<harness_default_partitioner>();
    ASSERT_WARNING((Harness::ConcurrencyTracker::PeakParallelism() > 1), "no parallel execution\n");
}

#include "tbb/task_scheduler_init.h"
#include "harness_cpu.h"
#include "test_partitioner.h"

namespace interaction_with_range_and_partitioner {

// Test checks compatibility of parallel_reduce algorithm with various range implementations

void test() {
    using namespace test_partitioner_utils::interaction_with_range_and_partitioner;

    test_partitioner_utils::SimpleReduceBody body;
    tbb::affinity_partitioner ap;

    parallel_reduce(Range1(/*assert_in_split*/ true, /*assert_in_proportional_split*/ false), body, ap);
    parallel_reduce(Range2(true, false), body, ap);
    parallel_reduce(Range3(true, false), body, ap);
    parallel_reduce(Range4(false, true), body, ap);
    parallel_reduce(Range5(false, true), body, ap);
    parallel_reduce(Range6(false, true), body, ap);

    parallel_reduce(Range1(/*assert_in_split*/ true, /*assert_in_proportional_split*/ false),
                           body, tbb::static_partitioner());
    parallel_reduce(Range2(true, false), body, tbb::static_partitioner());
    parallel_reduce(Range3(true, false), body, tbb::static_partitioner());
    parallel_reduce(Range4(false, true), body, tbb::static_partitioner());
    parallel_reduce(Range5(false, true), body, tbb::static_partitioner());
    parallel_reduce(Range6(false, true), body, tbb::static_partitioner());

    parallel_reduce(Range1(/*assert_in_split*/ false, /*assert_in_proportional_split*/ true),
                           body, tbb::simple_partitioner());
    parallel_reduce(Range2(false, true), body, tbb::simple_partitioner());
    parallel_reduce(Range3(false, true), body, tbb::simple_partitioner());
    parallel_reduce(Range4(false, true), body, tbb::simple_partitioner());
    parallel_reduce(Range5(false, true), body, tbb::simple_partitioner());
    parallel_reduce(Range6(false, true), body, tbb::simple_partitioner());

    parallel_reduce(Range1(/*assert_in_split*/ false, /*assert_in_proportional_split*/ true),
                           body, tbb::auto_partitioner());
    parallel_reduce(Range2(false, true), body, tbb::auto_partitioner());
    parallel_reduce(Range3(false, true), body, tbb::auto_partitioner());
    parallel_reduce(Range4(false, true), body, tbb::auto_partitioner());
    parallel_reduce(Range5(false, true), body, tbb::auto_partitioner());
    parallel_reduce(Range6(false, true), body, tbb::auto_partitioner());

    parallel_deterministic_reduce(Range1(/*assert_in_split*/true, /*assert_in_proportional_split*/ false),
                                         body, tbb::static_partitioner());
    parallel_deterministic_reduce(Range2(true, false), body, tbb::static_partitioner());
    parallel_deterministic_reduce(Range3(true, false), body, tbb::static_partitioner());
    parallel_deterministic_reduce(Range4(false, true), body, tbb::static_partitioner());
    parallel_deterministic_reduce(Range5(false, true), body, tbb::static_partitioner());
    parallel_deterministic_reduce(Range6(false, true), body, tbb::static_partitioner());

    parallel_deterministic_reduce(Range1(/*assert_in_split*/false, /*assert_in_proportional_split*/ true),
                                         body, tbb::simple_partitioner());
    parallel_deterministic_reduce(Range2(false, true), body, tbb::simple_partitioner());
    parallel_deterministic_reduce(Range3(false, true), body, tbb::simple_partitioner());
    parallel_deterministic_reduce(Range4(false, true), body, tbb::simple_partitioner());
    parallel_deterministic_reduce(Range5(false, true), body, tbb::simple_partitioner());
    parallel_deterministic_reduce(Range6(false, true), body, tbb::simple_partitioner());
}

} // interaction_with_range_and_partitioner

int TestMain () {
    TestUnsupportedPartitioners();
    if( MinThread<0 ) {
        REPORT("Usage: nthread must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init( p );
        Flog(p);
        ParallelSum();
        if ( p>=2 )
            TestDeterministicReduction();
        // Test that all workers sleep when no work
        TestCPUUserTime(p);
    }
    interaction_with_range_and_partitioner::test();
    return Harness::Done;
}
