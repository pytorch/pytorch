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

// Test for function template parallel_for.h

// These features are pure additions and thus can be always "on" in the test
#define TBB_PREVIEW_SERIAL_SUBSET 1
#include "harness_defs.h"

#if _MSC_VER
#pragma warning (push)
#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    // Suppress pointless "unreachable code" warning.
    #pragma warning (disable: 4702)
#endif
#if defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

#define _SCL_SECURE_NO_WARNINGS
#endif //#if _MSC_VER

#include "harness_defs.h"
#include "tbb/parallel_for.h"
#include "tbb/atomic.h"
#include "harness_assert.h"
#include "harness.h"

static tbb::atomic<int> FooBodyCount;

//! A range object whose only public members are those required by the Range concept.
template<size_t Pad>
class FooRange {
    //! Start of range
    int start;

    //! Size of range
    int size;
    FooRange( int start_, int size_ ) : start(start_), size(size_) {
        zero_fill<char>(pad, Pad);
        pad[Pad-1] = 'x';
    }
    template<typename Flavor_, size_t Pad_> friend void Flog( int nthread );
    template<size_t Pad_> friend class FooBody;
    void operator&();

    char pad[Pad];
public:
    bool empty() const {return size==0;}
    bool is_divisible() const {return size>1;}
    FooRange( FooRange& original, tbb::split ) : size(original.size/2) {
        original.size -= size;
        start = original.start+original.size;
        ASSERT( original.pad[Pad-1]=='x', NULL );
        pad[Pad-1] = 'x';
    }
};

//! A range object whose only public members are those required by the parallel_for.h body concept.
template<size_t Pad>
class FooBody {
    static const int LIVE = 0x1234;
    tbb::atomic<int>* array;
    int state;
    friend class FooRange<Pad>;
    template<typename Flavor_, size_t Pad_> friend void Flog( int nthread );
    FooBody( tbb::atomic<int>* array_ ) : array(array_), state(LIVE) {}
public:
    ~FooBody() {
        --FooBodyCount;
        for( size_t i=0; i<sizeof(*this); ++i )
            reinterpret_cast<char*>(this)[i] = -1;
    }
    //! Copy constructor
    FooBody( const FooBody& other ) : array(other.array), state(other.state) {
        ++FooBodyCount;
        ASSERT( state==LIVE, NULL );
    }
    void operator()( FooRange<Pad>& r ) const {
        for( int k=0; k<r.size; ++k ) {
            const int i = array[r.start+k]++;
            ASSERT( i==0, NULL );
        }
    }
};

#include "tbb/tick_count.h"

static const int N = 500;
static tbb::atomic<int> Array[N];

struct serial_tag {};
struct parallel_tag {};
struct empty_partitioner_tag {};

template <typename Flavor, typename Partitioner, typename Range, typename Body>
struct Invoker;

#if TBB_PREVIEW_SERIAL_SUBSET
template <typename Range, typename Body>
struct Invoker<serial_tag, empty_partitioner_tag, Range, Body> {
    void operator()( const Range& r, const Body& body, empty_partitioner_tag& ) {
        tbb::serial:: parallel_for( r, body );
    }
};
template <typename Partitioner, typename Range, typename Body>
struct Invoker<serial_tag, Partitioner, Range, Body> {
    void operator()( const Range& r, const Body& body, Partitioner& p ) {
        tbb::serial:: parallel_for( r, body, p );
    }
};
#endif

template <typename Range, typename Body>
struct Invoker<parallel_tag, empty_partitioner_tag, Range, Body> {
    void operator()( const Range& r, const Body& body, empty_partitioner_tag& ) {
        tbb:: parallel_for( r, body );
    }
};

template <typename Partitioner, typename Range, typename Body>
struct Invoker<parallel_tag, Partitioner, Range, Body> {
    void operator()( const Range& r, const Body& body, Partitioner& p ) {
        tbb:: parallel_for( r, body, p );
    }
};

template <typename Flavor, typename Partitioner, typename T, typename Body>
struct InvokerStep;

#if TBB_PREVIEW_SERIAL_SUBSET
template <typename T, typename Body>
struct InvokerStep<serial_tag, empty_partitioner_tag, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, empty_partitioner_tag& ) {
        tbb::serial:: parallel_for( first, last, f );
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, empty_partitioner_tag& ) {
        tbb::serial:: parallel_for( first, last, step, f );
    }
};

template <typename Partitioner, typename T, typename Body>
struct InvokerStep<serial_tag, Partitioner, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, Partitioner& p ) {
        tbb::serial:: parallel_for( first, last, f, p);
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, Partitioner& p ) {
        tbb::serial:: parallel_for( first, last, step, f, p );
    }
};
#endif

template <typename T, typename Body>
struct InvokerStep<parallel_tag, empty_partitioner_tag, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, empty_partitioner_tag& ) {
        tbb:: parallel_for( first, last, f );
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, empty_partitioner_tag& ) {
        tbb:: parallel_for( first, last, step, f );
    }
};

template <typename Partitioner, typename T, typename Body>
struct InvokerStep<parallel_tag, Partitioner, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, Partitioner& p ) {
        tbb:: parallel_for( first, last, f, p );
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, Partitioner& p ) {
        tbb:: parallel_for( first, last, step, f, p );
    }
};

template<typename Flavor, size_t Pad>
void Flog( int nthread ) {
    tbb::tick_count T0 = tbb::tick_count::now();
    for( int i=0; i<N; ++i ) {
        for ( int mode = 0; mode < 4; ++mode) {
            FooRange<Pad> r( 0, i );
            const FooRange<Pad> rc = r;
            FooBody<Pad> f( Array );
            const FooBody<Pad> fc = f;
            memset( Array, 0, sizeof(Array) );
            FooBodyCount = 1;
            switch (mode) {
            case 0: {
                empty_partitioner_tag p;
                Invoker< Flavor, empty_partitioner_tag, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, p );
            }
                break;
            case 1: {
                Invoker< Flavor, const tbb::simple_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, tbb::simple_partitioner() );
            }
                break;
            case 2: {
                Invoker< Flavor, const tbb::auto_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, tbb::auto_partitioner() );
            }
                break;
            case 3: {
                static tbb::affinity_partitioner affinity;
                Invoker< Flavor, tbb::affinity_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, affinity );
            }
                break;
            }
            for( int j=0; j<i; ++j )
                ASSERT( Array[j]==1, NULL );
            for( int j=i; j<N; ++j )
                ASSERT( Array[j]==0, NULL );
            ASSERT( FooBodyCount==1, NULL );
        }
    }
    tbb::tick_count T1 = tbb::tick_count::now();
    REMARK("time=%g\tnthread=%d\tpad=%d\n",(T1-T0).seconds(),nthread,int(Pad));
}

// Testing parallel_for with step support
const size_t PFOR_BUFFER_TEST_SIZE = 1024;
// test_buffer has some extra items beyond its right bound
const size_t PFOR_BUFFER_ACTUAL_SIZE = PFOR_BUFFER_TEST_SIZE + 1024;
size_t pfor_buffer[PFOR_BUFFER_ACTUAL_SIZE];

template<typename T>
class TestFunctor{
public:
    void operator ()(T index) const {
        pfor_buffer[index]++;
    }
};

#include <stdexcept> // std::invalid_argument

template <typename Flavor, typename T, typename Partitioner>
void TestParallelForWithStepSupportHelper(Partitioner& p)
{
    const T pfor_buffer_test_size = static_cast<T>(PFOR_BUFFER_TEST_SIZE);
    const T pfor_buffer_actual_size = static_cast<T>(PFOR_BUFFER_ACTUAL_SIZE);
    // Testing parallel_for with different step values
    InvokerStep< Flavor, Partitioner, T, TestFunctor<T> > invoke_for;
    for (T begin = 0; begin < pfor_buffer_test_size - 1; begin += pfor_buffer_test_size / 10 + 1) {
        T step;
        for (step = 1; step < pfor_buffer_test_size; step++) {
            memset(pfor_buffer, 0, pfor_buffer_actual_size * sizeof(size_t));
            if (step == 1){
                invoke_for(begin, pfor_buffer_test_size, TestFunctor<T>(), p);
            } else {
                invoke_for(begin, pfor_buffer_test_size, step, TestFunctor<T>(), p);
            }
            // Verifying that parallel_for processed all items it should
            for (T i = begin; i < pfor_buffer_test_size; i = i + step) {
                ASSERT(pfor_buffer[i] == 1, "parallel_for didn't process all required elements");
                pfor_buffer[i] = 0;
            }
            // Verifying that no extra items were processed and right bound of array wasn't crossed
            for (T i = 0; i < pfor_buffer_actual_size; i++) {
                ASSERT(pfor_buffer[i] == 0, "parallel_for processed an extra element");
            }
        }
    }
}

template <typename Flavor, typename T>
void TestParallelForWithStepSupport()
{
    static tbb::affinity_partitioner affinity_p;
    tbb::auto_partitioner auto_p;
    tbb::simple_partitioner simple_p;
    empty_partitioner_tag p;

    // Try out all partitioner combinations
    TestParallelForWithStepSupportHelper< Flavor,T,empty_partitioner_tag >(p);
    TestParallelForWithStepSupportHelper< Flavor,T,const tbb::auto_partitioner >(auto_p);
    TestParallelForWithStepSupportHelper< Flavor,T,const tbb::simple_partitioner >(simple_p);
    TestParallelForWithStepSupportHelper< Flavor,T,tbb::affinity_partitioner >(affinity_p);

    // Testing some corner cases
    tbb::parallel_for(static_cast<T>(2), static_cast<T>(1), static_cast<T>(1), TestFunctor<T>());
#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    try{
        tbb::parallel_for(static_cast<T>(1), static_cast<T>(100), static_cast<T>(0), TestFunctor<T>());  // should cause std::invalid_argument
    }catch(std::invalid_argument){
        return;
    }
    catch ( ... ) {
        ASSERT ( __TBB_EXCEPTION_TYPE_INFO_BROKEN, "Unrecognized exception. std::invalid_argument is expected" );
    }
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
}

#if __TBB_TASK_GROUP_CONTEXT
// Exception support test
#define HARNESS_EH_SIMPLE_MODE 1
#include "tbb/tbb_exception.h"
#include "harness_eh.h"

#if TBB_USE_EXCEPTIONS
class test_functor_with_exception {
public:
    void operator ()(size_t) const { ThrowTestException(); }
};

void TestExceptionsSupport() {
    REMARK (__FUNCTION__);
    { // Tests version with a step provided
        ResetEhGlobals();
        TRY();
            tbb::parallel_for((size_t)0, (size_t)PFOR_BUFFER_TEST_SIZE, (size_t)1, test_functor_with_exception());
        CATCH_AND_ASSERT();
    }
    { // Tests version without a step
        ResetEhGlobals();
        TRY();
            tbb::parallel_for((size_t)0, (size_t)PFOR_BUFFER_TEST_SIZE, test_functor_with_exception());
        CATCH_AND_ASSERT();
    }
}
#endif /* TBB_USE_EXCEPTIONS */

// Cancellation support test
class functor_to_cancel {
public:
    void operator()(size_t) const {
        ++g_CurExecuted;
        CancellatorTask::WaitUntilReady();
    }
};

size_t g_worker_task_step = 0;

class my_worker_pfor_step_task : public tbb::task
{
    tbb::task_group_context &my_ctx;

    tbb::task* execute () __TBB_override {
        if (g_worker_task_step == 0){
            tbb::parallel_for((size_t)0, (size_t)PFOR_BUFFER_TEST_SIZE, functor_to_cancel(), my_ctx);
        }else{
            tbb::parallel_for((size_t)0, (size_t)PFOR_BUFFER_TEST_SIZE, g_worker_task_step, functor_to_cancel(), my_ctx);
        }
        return NULL;
    }
public:
    my_worker_pfor_step_task ( tbb::task_group_context &context_) : my_ctx(context_) { }
};

void TestCancellation()
{
    // tests version without a step
    g_worker_task_step = 0;
    ResetEhGlobals();
    RunCancellationTest<my_worker_pfor_step_task, CancellatorTask>();

    // tests version with step
    g_worker_task_step = 1;
    ResetEhGlobals();
    RunCancellationTest<my_worker_pfor_step_task, CancellatorTask>();
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

#include "harness_m128.h"

#if (HAVE_m128 || HAVE_m256) && !__TBB_SSE_STACK_ALIGNMENT_BROKEN
template<typename ClassWithVectorType>
struct SSE_Functor {
    ClassWithVectorType* Src, * Dst;
    SSE_Functor( ClassWithVectorType* src, ClassWithVectorType* dst ) : Src(src), Dst(dst) {}

    void operator()( tbb::blocked_range<int>& r ) const {
        for( int i=r.begin(); i!=r.end(); ++i )
            Dst[i] = Src[i];
    }
};

//! Test that parallel_for works with stack-allocated __m128
template<typename ClassWithVectorType>
void TestVectorTypes() {
    ClassWithVectorType Array1[N], Array2[N];
    for( int i=0; i<N; ++i ) {
        // VC8 does not properly align a temporary value; to work around, use explicit variable
        ClassWithVectorType foo(i);
        Array1[i] = foo;
    }
    tbb::parallel_for( tbb::blocked_range<int>(0,N), SSE_Functor<ClassWithVectorType>(Array1, Array2) );
    for( int i=0; i<N; ++i ) {
        ClassWithVectorType foo(i);
        ASSERT( Array2[i]==foo, NULL ) ;
    }
}
#endif /* HAVE_m128 || HAVE_m256 */

#include <vector>
#include <sstream>
#include <tbb/blocked_range.h>

struct TestSimplePartitionerStabilityFunctor:NoAssign{
  std::vector<int> & ranges;
  TestSimplePartitionerStabilityFunctor(std::vector<int> & theRanges):ranges(theRanges){}
  void operator()(tbb::blocked_range<size_t>& r)const{
      ranges.at(r.begin())=true;
  }
};
void TestSimplePartitionerStability(){
    const std::size_t repeat_count= 10;
    const std::size_t rangeToSplitSize=1000000;
    const std::size_t grainsizeStep=rangeToSplitSize/repeat_count;
    typedef TestSimplePartitionerStabilityFunctor FunctorType;

    for (std::size_t i=0 , grainsize=grainsizeStep; i<repeat_count;i++, grainsize+=grainsizeStep){
        std::vector<int> firstSeries(rangeToSplitSize,0);
        std::vector<int> secondSeries(rangeToSplitSize,0);

        tbb::parallel_for(tbb::blocked_range<size_t>(0,rangeToSplitSize,grainsize),FunctorType(firstSeries),tbb::simple_partitioner());
        tbb::parallel_for(tbb::blocked_range<size_t>(0,rangeToSplitSize,grainsize),FunctorType(secondSeries),tbb::simple_partitioner());
        std::stringstream str; str<<i;
        ASSERT(firstSeries==secondSeries,("splitting range with tbb::simple_partitioner must be reproducible; i=" +str.str()).c_str() );
    }
}
#include <cstdio>
#include "tbb/task_scheduler_init.h"
#include "harness_cpu.h"
#include "harness_barrier.h"
#include "test_partitioner.h"

namespace interaction_with_range_and_partitioner {

// Test checks compatibility of parallel_for algorithm with various range implementations

void test() {
    using namespace test_partitioner_utils::interaction_with_range_and_partitioner;

    test_partitioner_utils::SimpleBody b;
    tbb::affinity_partitioner ap;

    parallel_for(Range1(true, false), b, ap);
    parallel_for(Range2(true, false), b, ap);
    parallel_for(Range3(true, false), b, ap);
    parallel_for(Range4(false, true), b, ap);
    parallel_for(Range5(false, true), b, ap);
    parallel_for(Range6(false, true), b, ap);

    parallel_for(Range1(false, true), b, tbb::simple_partitioner());
    parallel_for(Range2(false, true), b, tbb::simple_partitioner());
    parallel_for(Range3(false, true), b, tbb::simple_partitioner());
    parallel_for(Range4(false, true), b, tbb::simple_partitioner());
    parallel_for(Range5(false, true), b, tbb::simple_partitioner());
    parallel_for(Range6(false, true), b, tbb::simple_partitioner());

    parallel_for(Range1(false, true), b, tbb::auto_partitioner());
    parallel_for(Range2(false, true), b, tbb::auto_partitioner());
    parallel_for(Range3(false, true), b, tbb::auto_partitioner());
    parallel_for(Range4(false, true), b, tbb::auto_partitioner());
    parallel_for(Range5(false, true), b, tbb::auto_partitioner());
    parallel_for(Range6(false, true), b, tbb::auto_partitioner());
}

} // namespace interaction_with_range_and_partitioner

namespace various_range_implementations {

using namespace test_partitioner_utils;
using namespace test_partitioner_utils::TestRanges;

// Body ensures that initial work distribution is done uniformly through affinity mechanism and not
// through work stealing
class Body {
    Harness::SpinBarrier &m_sb;
public:
    Body(Harness::SpinBarrier& sb) : m_sb(sb) { }
    Body(Body& b, tbb::split) : m_sb(b.m_sb) { }
    Body& operator =(const Body&) { return *this; }
    template <typename Range>
    void operator()(Range& r) const {
        REMARK("Executing range [%lu, %lu)\n", r.begin(), r.end());
        m_sb.timed_wait(10); // waiting for all threads
    }
};

namespace correctness {

/* Testing only correctness (that is parallel_for does not hang) */
template <typename RangeType, bool /* feedback */, bool ensure_non_emptiness>
void test() {
    static const int thread_num = tbb::task_scheduler_init::default_num_threads();
    RangeType range( 0, thread_num, NULL, false, ensure_non_emptiness );
    tbb::affinity_partitioner ap;
    tbb::parallel_for( range, SimpleBody(), ap );
}

} // namespace correctness

namespace uniform_distribution {

/* Body of parallel_for algorithm would hang if non-uniform work distribution happened  */
template <typename RangeType, bool feedback, bool ensure_non_emptiness>
void test() {
    static const int thread_num = tbb::task_scheduler_init::default_num_threads();
    Harness::SpinBarrier sb( thread_num );
    RangeType range(0, thread_num, NULL, feedback, ensure_non_emptiness);
    const Body sync_body( sb );
    tbb::affinity_partitioner ap;
    tbb::parallel_for( range, sync_body, ap );
    tbb::parallel_for( range, sync_body, tbb::static_partitioner() );
}

} // namespace uniform_distribution

void test() {
    const bool provide_feedback = __TBB_ENABLE_RANGE_FEEDBACK;
    const bool ensure_non_empty_range = true;

    // BlockedRange does not take into account feedback and non-emptiness settings but uses the
    // tbb::blocked_range implementation
    uniform_distribution::test<BlockedRange, !provide_feedback, !ensure_non_empty_range>();

#if __TBB_ENABLE_RANGE_FEEDBACK
    using uniform_distribution::test; // if feedback is enabled ensure uniform work distribution
#else
    using correctness::test;
#endif

    {
        test<RoundedDownRange, provide_feedback, ensure_non_empty_range>();
        test<RoundedDownRange, provide_feedback, !ensure_non_empty_range>();
#if __TBB_ENABLE_RANGE_FEEDBACK && !__TBB_MIC_NATIVE
        // due to fast division algorithm on MIC
        test<RoundedDownRange, !provide_feedback, ensure_non_empty_range>();
        test<RoundedDownRange, !provide_feedback, !ensure_non_empty_range>();
#endif
    }

    {
        test<RoundedUpRange, provide_feedback, ensure_non_empty_range>();
        test<RoundedUpRange, provide_feedback, !ensure_non_empty_range>();
#if __TBB_ENABLE_RANGE_FEEDBACK && !__TBB_MIC_NATIVE
        // due to fast division algorithm on MIC
        test<RoundedUpRange, !provide_feedback, ensure_non_empty_range>();
        test<RoundedUpRange, !provide_feedback, !ensure_non_empty_range>();
#endif
    }

    // Testing that parallel_for algorithm works with such weird ranges
    correctness::test<Range1_2, /* provide_feedback= */ false, !ensure_non_empty_range>();
    correctness::test<Range1_999, /* provide_feedback= */ false, !ensure_non_empty_range>();
    correctness::test<Range999_1, /* provide_feedback= */ false, !ensure_non_empty_range>();

    // The following ranges do not comply with the proportion suggested by partitioner. Therefore
    // they have to provide the proportion in which they were actually split back to partitioner and
    // ensure theirs non-emptiness
    test<Range1_2, provide_feedback, ensure_non_empty_range>();
    test<Range1_999, provide_feedback, ensure_non_empty_range>();
    test<Range999_1, provide_feedback, ensure_non_empty_range>();
}

} // namespace various_range_implementations

#include <map>
#include <utility>
#include "tbb/task_arena.h"
#include "tbb/enumerable_thread_specific.h"

namespace parallel_for_within_task_arena {

using namespace test_partitioner_utils::TestRanges;
using tbb::split;
using tbb::proportional_split;

class BlockedRangeWhitebox;

typedef std::pair<size_t, size_t> range_borders;
typedef std::multimap<BlockedRangeWhitebox*, range_borders> MapType;
typedef tbb::enumerable_thread_specific<MapType> ETSType;
ETSType ets;

class BlockedRangeWhitebox : public BlockedRange {
public:
    static const bool is_splittable_in_proportion = true;
    BlockedRangeWhitebox(size_t _begin, size_t _end)
        : BlockedRange(_begin, _end, NULL, false, false) { }

    BlockedRangeWhitebox(BlockedRangeWhitebox& r, proportional_split& p)
        :BlockedRange(r, p) {
        update_ets(r);
        update_ets(*this);
    }

    BlockedRangeWhitebox(BlockedRangeWhitebox& r, split)
        :BlockedRange(r, split()) { }

    void update_ets(BlockedRangeWhitebox& range) {
        std::pair<MapType::iterator, MapType::iterator> equal_range = ets.local().equal_range(&range);
        for (MapType::iterator it = equal_range.first; it != equal_range.second;++it) {
            if (it->second.first <= range.begin() && range.end() <= it->second.second) {
                ASSERT(!(it->second.first == range.begin() && it->second.second == range.end()), "Only one border of the range should be equal to the original");
                it->second.first = range.begin();
                it->second.second = range.end();
                return;
            }
        }
        ets.local().insert(std::make_pair<BlockedRangeWhitebox*, range_borders>(&range, range_borders(range.begin(), range.end())));
    }
};

template <typename Partitioner>
struct ArenaBody {
    size_t range_begin;
    size_t range_end;

    ArenaBody(size_t _range_begin, size_t _range_end)
        :range_begin(_range_begin), range_end(_range_end) { }

    void operator()() const {
        Partitioner my_partitioner;
        tbb::parallel_for(BlockedRangeWhitebox(range_begin, range_end), test_partitioner_utils::SimpleBody(), my_partitioner);
    }
};

struct CombineBody {
    MapType operator()(MapType x, const MapType& y) const {
        x.insert(y.begin(), y.end());
        for (MapType::iterator it = x.begin(); it != x.end();++it)
            for (MapType::iterator internal_it = x.begin(); internal_it != x.end(); ++internal_it) {
                if (it != internal_it && internal_it->second.first <= it->second.first && it->second.second <= internal_it->second.second) {
                    x.erase(internal_it);
                    break;
                }
            }
        return x;
    }
};

range_borders combine_range(const MapType& map) {
    range_borders result_range = map.begin()->second;
    for (MapType::const_iterator it = map.begin(); it != map.end(); it++)
        result_range = range_borders((std::min)(result_range.first, it->second.first), (std::max)(result_range.second, it->second.second));
    return result_range;
}

template <typename Partitioner>
void test_body() {
    for (unsigned int num_threads = tbb::tbb_thread::hardware_concurrency() / 4 + 1; num_threads < tbb::tbb_thread::hardware_concurrency(); num_threads *= 2)
        for (size_t range_begin = 0, range_end = num_threads * 10 - 1, i = 0; i < 3; range_begin += num_threads, range_end += num_threads + 1, ++i) {
            ets = ETSType(MapType());
            tbb::task_arena limited(num_threads);
            limited.execute(ArenaBody<Partitioner>(range_begin, range_end));
            MapType combined_map = ets.combine(CombineBody());
            range_borders result_borders = combine_range(combined_map);
            ASSERT(result_borders.first == range_begin, "Restored range begin does not match initial one");
            ASSERT(result_borders.second == range_end, "Restored range end does not match initial one");
            ASSERT((combined_map.size() == num_threads), "Incorrect number or post-proportional split ranges");
            size_t expected_size = (range_end - range_begin) / num_threads;
            for (MapType::iterator it = combined_map.begin(); it != combined_map.end(); ++it) {
                size_t size = it->second.second - it->second.first;
                ASSERT((size == expected_size || size == expected_size + 1), "Incorrect post-proportional range size");
            }
        }

}

void test() {
    test_body<tbb::affinity_partitioner>();
    test_body<tbb::static_partitioner>();
}

} // namespace parallel_for_within_task_arena

int TestMain () {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
        if( p>0 ) {
            tbb::task_scheduler_init init( p );
            Flog<parallel_tag,1>(p);
            Flog<parallel_tag,10>(p);
            Flog<parallel_tag,100>(p);
            Flog<parallel_tag,1000>(p);
            Flog<parallel_tag,10000>(p);

            // Testing with different integer types
            TestParallelForWithStepSupport<parallel_tag,short>();
            TestParallelForWithStepSupport<parallel_tag,unsigned short>();
            TestParallelForWithStepSupport<parallel_tag,int>();
            TestParallelForWithStepSupport<parallel_tag,unsigned int>();
            TestParallelForWithStepSupport<parallel_tag,long>();
            TestParallelForWithStepSupport<parallel_tag,unsigned long>();
            TestParallelForWithStepSupport<parallel_tag,long long>();
            TestParallelForWithStepSupport<parallel_tag,unsigned long long>();
            TestParallelForWithStepSupport<parallel_tag,size_t>();

#if TBB_PREVIEW_SERIAL_SUBSET
            // This is for testing serial implementation.
            if( p == MaxThread ) {
                Flog<serial_tag,1>(p);
                Flog<serial_tag,10>(p);
                Flog<serial_tag,100>(p);
                TestParallelForWithStepSupport<serial_tag,short>();
                TestParallelForWithStepSupport<serial_tag,unsigned short>();
                TestParallelForWithStepSupport<serial_tag,int>();
                TestParallelForWithStepSupport<serial_tag,unsigned int>();
                TestParallelForWithStepSupport<serial_tag,long>();
                TestParallelForWithStepSupport<serial_tag,unsigned long>();
                TestParallelForWithStepSupport<serial_tag,long long>();
                TestParallelForWithStepSupport<serial_tag,unsigned long long>();
                TestParallelForWithStepSupport<serial_tag,size_t>();
            }
#endif

#if TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
            TestExceptionsSupport();
#endif /* TBB_USE_EXCEPTIONS && !__TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN */
#if __TBB_TASK_GROUP_CONTEXT
            if ( p > 1 )
                TestCancellation();
#endif /* __TBB_TASK_GROUP_CONTEXT */
#if !__TBB_SSE_STACK_ALIGNMENT_BROKEN
    #if HAVE_m128
            TestVectorTypes<ClassWithSSE>();
    #endif
    #if HAVE_m256
            if (have_AVX()) TestVectorTypes<ClassWithAVX>();
    #endif
#endif /*!__TBB_SSE_STACK_ALIGNMENT_BROKEN*/
            // Test that all workers sleep when no work
            TestCPUUserTime(p);
            TestSimplePartitionerStability();
        }
    }
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception handling tests are skipped.\n");
#endif
#if (HAVE_m128 || HAVE_m256) && __TBB_SSE_STACK_ALIGNMENT_BROKEN
    REPORT("Known issue: stack alignment for SIMD instructions not tested.\n");
#endif

    various_range_implementations::test();
    interaction_with_range_and_partitioner::test();
    parallel_for_within_task_arena::test();
    return Harness::Done;
}

#if _MSC_VER
#pragma warning (pop)
#endif
