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

#define HARNESS_DEFAULT_MIN_THREADS 0
#define HARNESS_DEFAULT_MAX_THREADS 4

#define __TBB_EXTRA_DEBUG 1 // for concurrent_hash_map
#include "tbb/combinable.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"
#include "tbb/tbb_allocator.h"
#include "tbb/tbb_thread.h"

#include <cstring>
#include <vector>
#include <utility>

#include "harness_assert.h"
#include "harness.h"
#include "test_container_move_support.h"

#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

static tbb::atomic<int> construction_counter;
static tbb::atomic<int> destruction_counter;

const int REPETITIONS = 10;
const int N = 100000;
const double EXPECTED_SUM = (REPETITIONS + 1) * N;

//
// A minimal class
// Define: default and copy constructor, and allow implicit operator&
// also operator=
//

class minimal {
private:
    int my_value;
public:
    minimal(int val=0) : my_value(val) { ++construction_counter; }
    minimal( const minimal &m ) : my_value(m.my_value) { ++construction_counter; }
    minimal& operator=(const minimal& other) { my_value = other.my_value; return *this; }
    minimal& operator+=(const minimal& other) { my_value += other.my_value; return *this; }
    operator int() const { return my_value; }
    ~minimal() { ++destruction_counter; }
    void set_value( const int i ) { my_value = i; }
    int value( ) const { return my_value; }
};

//// functors for initialization and combine

template <typename T>
struct FunctorAddFinit {
    T operator()() { return 0; }
};

template <typename T>
struct FunctorAddFinit7 {
    T operator()() { return 7; }
};

template <typename T>
struct FunctorAddCombine {
    T operator()(T left, T right ) const {
        return left + right;
    }
};

template <typename T>
struct FunctorAddCombineRef {
    T operator()(const T& left, const T& right ) const {
        return left + right;
    }
};

template <typename T>
T my_combine( T left, T right) { return left + right; }

template <typename T>
T my_combine_ref( const T &left, const T &right) { return left + right; }

template <typename T>
class CombineEachHelper {
public:
    CombineEachHelper(T& _result) : my_result(_result) {}
    void operator()(const T& new_bit) { my_result +=  new_bit; }
    CombineEachHelper& operator=(const CombineEachHelper& other) {
        my_result =  other;
        return *this;
    }
private:
    T& my_result;
};

template <typename T>
class CombineEachHelperCnt {
public:
    CombineEachHelperCnt(T& _result, int& _nbuckets) : my_result(_result), nBuckets(_nbuckets) {}
    void operator()(const T& new_bit) { my_result +=  new_bit; ++nBuckets; }
    CombineEachHelperCnt& operator=(const CombineEachHelperCnt& other) {
        my_result =  other.my_result;
        nBuckets = other.nBuckets;
        return *this;
    }
private:
    T& my_result;
    int& nBuckets;
};

template <typename T>
class CombineEachVectorHelper {
public:
    typedef std::vector<T, tbb::tbb_allocator<T> > ContainerType;
    CombineEachVectorHelper(T& _result) : my_result(_result) { }
    void operator()(const ContainerType& new_bit) {
        for(typename ContainerType::const_iterator ci = new_bit.begin(); ci != new_bit.end(); ++ci) {
            my_result +=  *ci;
        }
    }
    CombineEachVectorHelper& operator=(const CombineEachVectorHelper& other) {
        my_result=other.my_result;
        return *this;
    }

private:
    T& my_result;
};

//// end functors

// parallel body with a test for first access
template <typename T>
class ParallelScalarBody: NoAssign {

    tbb::combinable<T> &sums;

public:

    ParallelScalarBody ( tbb::combinable<T> &_sums ) : sums(_sums) { }

    void operator()( const tbb::blocked_range<int> &r ) const {
        for (int i = r.begin(); i != r.end(); ++i) {
            bool was_there;
            T& my_local = sums.local(was_there);
            if(!was_there) my_local = 0;
             my_local +=  1 ;
        }
    }

};

// parallel body with no test for first access
template <typename T>
class ParallelScalarBodyNoInit: NoAssign {

    tbb::combinable<T> &sums;

public:

    ParallelScalarBodyNoInit ( tbb::combinable<T> &_sums ) : sums(_sums) { }

    void operator()( const tbb::blocked_range<int> &r ) const {
        for (int i = r.begin(); i != r.end(); ++i) {
             sums.local() +=  1 ;
        }
    }

};

template< typename T >
void RunParallelScalarTests(const char *test_name) {

    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    for (int p = MinThread; p <= MaxThread; ++p) {

        if (p == 0) continue;
        REMARK("  Testing parallel %s on %d thread(s)...\n", test_name, p);
        init.initialize(p);

        tbb::tick_count t0;
        T combine_sum(0);
        T combine_ref_sum(0);
        T combine_finit_sum(0);
        T combine_each_sum(0);
        T copy_construct_sum(0);
        T copy_assign_sum(0);
#if __TBB_ETS_USE_CPP11
        T move_construct_sum(0);
        T move_assign_sum(0);
#endif
        for (int t = -1; t < REPETITIONS; ++t) {
            if (Verbose && t == 0) t0 = tbb::tick_count::now();

            // test uninitialized parallel combinable
            tbb::combinable<T> sums;
            tbb::parallel_for( tbb::blocked_range<int>( 0, N, 10000 ), ParallelScalarBody<T>( sums ) );
            combine_sum += sums.combine(my_combine<T>);
            combine_ref_sum += sums.combine(my_combine_ref<T>);

            // test parallel combinable preinitialized with a functor that returns 0
            FunctorAddFinit<T> my_finit_decl;
            tbb::combinable<T> finit_combinable(my_finit_decl);
            tbb::parallel_for( tbb::blocked_range<int>( 0, N, 10000 ), ParallelScalarBodyNoInit<T>( finit_combinable ) );
            combine_finit_sum += finit_combinable.combine(my_combine<T>);

            // test another way of combining the elements using CombineEachHelper<T> functor
            CombineEachHelper<T> my_helper(combine_each_sum);
            sums.combine_each(my_helper);

            // test copy constructor for parallel combinable
            tbb::combinable<T> copy_constructed(sums);
            copy_construct_sum += copy_constructed.combine(my_combine<T>);

            // test copy assignment for uninitialized parallel combinable
            tbb::combinable<T> assigned;
            assigned = sums;
            copy_assign_sum += assigned.combine(my_combine<T>);

#if __TBB_ETS_USE_CPP11
            // test move constructor for parallel combinable
            tbb::combinable<T> moved1(std::move(sums));
            move_construct_sum += moved1.combine(my_combine<T>);

            // test move assignment for uninitialized parallel combinable
            tbb::combinable<T> moved2;
            moved2=std::move(finit_combinable);
            move_assign_sum += moved2.combine(my_combine<T>);
#endif
        }
        // Here and below comparison for equality of float numbers succeeds
        // as the rounding error doesn't accumulate and doesn't affect the comparison
        ASSERT( EXPECTED_SUM == combine_sum, NULL);
        ASSERT( EXPECTED_SUM == combine_ref_sum, NULL);
        ASSERT( EXPECTED_SUM == combine_finit_sum, NULL);
        ASSERT( EXPECTED_SUM == combine_each_sum, NULL);
        ASSERT( EXPECTED_SUM == copy_construct_sum, NULL);
        ASSERT( EXPECTED_SUM == copy_assign_sum, NULL);
#if __TBB_ETS_USE_CPP11
        ASSERT( EXPECTED_SUM == move_construct_sum, NULL);
        ASSERT( EXPECTED_SUM == move_assign_sum, NULL);
#endif
        REMARK("  done parallel %s, %d, %g, %g\n", test_name, p, static_cast<double>(combine_sum),
                                                      ( tbb::tick_count::now() - t0).seconds());
        init.terminate();
    }
}

template <typename T>
class ParallelVectorForBody: NoAssign {

    tbb::combinable< std::vector<T, tbb::tbb_allocator<T> > > &locals;

public:

    ParallelVectorForBody ( tbb::combinable< std::vector<T, tbb::tbb_allocator<T> > > &_locals ) : locals(_locals) { }

    void operator()( const tbb::blocked_range<int> &r ) const {
        T one = 1;

        for (int i = r.begin(); i < r.end(); ++i) {
            locals.local().push_back( one );
        }
    }

};

template< typename T >
void RunParallelVectorTests(const char *test_name) {

    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);

    typedef std::vector<T, tbb::tbb_allocator<T> > ContainerType;

    for (int p = MinThread; p <= MaxThread; ++p) {

        if (p == 0) continue;
        REMARK("  Testing parallel %s on %d thread(s)... \n", test_name, p);
        init.initialize(p);

        tbb::tick_count t0;
        T defaultConstructed_sum(0);
        T copyConstructed_sum(0);
        T copyAssigned_sum(0);
#if __TBB_ETS_USE_CPP11
        T moveConstructed_sum(0);
        T moveAssigned_sum(0);
#endif
        for (int t = -1; t < REPETITIONS; ++t) {
            if (Verbose && t == 0) t0 = tbb::tick_count::now();

            typedef typename tbb::combinable< ContainerType > CombinableType;

            // test uninitialized parallel combinable
            CombinableType vs;
            tbb::parallel_for( tbb::blocked_range<int> (0, N, 10000), ParallelVectorForBody<T>( vs ) );
            CombineEachVectorHelper<T> MyCombineEach(defaultConstructed_sum);
            vs.combine_each(MyCombineEach); // combine_each sums all elements of each vector into the result

            // test copy constructor for parallel combinable with vectors
            CombinableType vs2(vs);
            CombineEachVectorHelper<T> MyCombineEach2(copyConstructed_sum);
            vs2.combine_each(MyCombineEach2);

            // test copy assignment for uninitialized parallel combinable with vectors
            CombinableType vs3;
            vs3 = vs;
            CombineEachVectorHelper<T> MyCombineEach3(copyAssigned_sum);
            vs3.combine_each(MyCombineEach3);

#if __TBB_ETS_USE_CPP11
            // test move constructor for parallel combinable with vectors
            CombinableType vs4(std::move(vs2));
            CombineEachVectorHelper<T> MyCombineEach4(moveConstructed_sum);
            vs4.combine_each(MyCombineEach4);

            // test move assignment for uninitialized parallel combinable with vectors
            vs4=std::move(vs3);
            CombineEachVectorHelper<T> MyCombineEach5(moveAssigned_sum);
            vs4.combine_each(MyCombineEach5);
#endif
        }

        double ResultValue = defaultConstructed_sum;
        ASSERT( EXPECTED_SUM == ResultValue, NULL);
        ResultValue = copyConstructed_sum;
        ASSERT( EXPECTED_SUM == ResultValue, NULL);
        ResultValue = copyAssigned_sum;
        ASSERT( EXPECTED_SUM == ResultValue, NULL);
#if __TBB_ETS_USE_CPP11
        ResultValue = moveConstructed_sum;
        ASSERT( EXPECTED_SUM == ResultValue, NULL);
        ResultValue = moveAssigned_sum;
        ASSERT( EXPECTED_SUM == ResultValue, NULL);
#endif
        REMARK("  done parallel %s, %d, %g, %g\n", test_name, p, ResultValue, ( tbb::tick_count::now() - t0).seconds());
        init.terminate();
    }
}

void
RunParallelTests() {
    REMARK("Running RunParallelTests\n");
    RunParallelScalarTests<int>("int");
    RunParallelScalarTests<double>("double");
    RunParallelScalarTests<minimal>("minimal");
    RunParallelVectorTests<int>("std::vector<int, tbb::tbb_allocator<int> >");
    RunParallelVectorTests<double>("std::vector<double, tbb::tbb_allocator<double> >");
}

template <typename T>
void
RunAssignmentAndCopyConstructorTest(const char *test_name) {
    REMARK("  Testing assignment and copy construction for combinable<%s>...\n", test_name);

    // test creation with finit function (combine returns finit return value if no threads have created locals)
    FunctorAddFinit7<T> my_finit7_decl;
    tbb::combinable<T> create1(my_finit7_decl);
    ASSERT(7 == create1.combine(my_combine<T>), "Unexpected combine result for combinable object preinitialized with functor");

    // test copy construction with function initializer
    tbb::combinable<T> copy1(create1);
    ASSERT(7 == copy1.combine(my_combine<T>), "Unexpected combine result for copy-constructed combinable object");

    // test copy assignment with function initializer
    FunctorAddFinit<T> my_finit_decl;
    tbb::combinable<T> assign1(my_finit_decl);
    assign1 = create1;
    ASSERT(7 == assign1.combine(my_combine<T>), "Unexpected combine result for copy-assigned combinable object");

#if __TBB_ETS_USE_CPP11
    // test move construction with function initializer
    tbb::combinable<T> move1(std::move(create1));
    ASSERT(7 == move1.combine(my_combine<T>), "Unexpected combine result for move-constructed combinable object");

    // test move assignment with function initializer
    tbb::combinable<T> move2;
    move2=std::move(copy1);
    ASSERT(7 == move2.combine(my_combine<T>), "Unexpected combine result for move-assigned combinable object");
#endif

    REMARK("  done\n");

}

void
RunAssignmentAndCopyConstructorTests() {
    REMARK("Running assignment and copy constructor tests:\n");
    RunAssignmentAndCopyConstructorTest<int>("int");
    RunAssignmentAndCopyConstructorTest<double>("double");
    RunAssignmentAndCopyConstructorTest<minimal>("minimal");
}

void
RunMoveSemanticsForStateTrackableObjectTest() {
    REMARK("Testing move assignment and move construction for combinable<Harness::StateTrackable>...\n");

    tbb::combinable< Harness::StateTrackable<true> > create1;
    ASSERT(create1.local().state == Harness::StateTrackable<true>::DefaultInitialized,
           "Unexpected value in default combinable object");

    // Copy constructing of the new combinable causes copying of stored values
    tbb::combinable< Harness::StateTrackable<true> > copy1(create1);
    ASSERT(copy1.local().state == Harness::StateTrackable<true>::CopyInitialized,
           "Unexpected value in copy-constructed combinable object");

    // Copy assignment also causes copying of stored values
    tbb::combinable< Harness::StateTrackable<true> > copy2;
    ASSERT(copy2.local().state == Harness::StateTrackable<true>::DefaultInitialized,
           "Unexpected value in default combinable object");
    copy2=create1;
    ASSERT(copy2.local().state == Harness::StateTrackable<true>::CopyInitialized,
           "Unexpected value in copy-assigned combinable object");

#if __TBB_ETS_USE_CPP11
    // Store some marked values in the initial combinable object
    create1.local().state = Harness::StateTrackableBase::Unspecified;

    // Move constructing of the new combinable must not cause copying of stored values
    tbb::combinable< Harness::StateTrackable<true> > move1(std::move(create1));
    ASSERT(move1.local().state == Harness::StateTrackableBase::Unspecified, "Unexpected value in move-constructed combinable object");

    // Move assignment must not cause copying of stored values
    copy1=std::move(move1);
    ASSERT(copy1.local().state == Harness::StateTrackableBase::Unspecified, "Unexpected value in move-assigned combinable object");

    // Make the stored values valid again in order to delete StateTrackable object correctly
    copy1.local().state = Harness::StateTrackable<true>::MoveAssigned;
#endif

    REMARK("done\n");
}

#include "harness_barrier.h"

Harness::SpinBarrier sBarrier;

struct Body : NoAssign {
    tbb::combinable<int>* locals;
    const int nthread;
    const int nIters;
    Body( int nthread_, int niters_ ) : nthread(nthread_), nIters(niters_) { sBarrier.initialize(nthread_); }

    void operator()(int thread_id ) const {
        bool existed;
        sBarrier.wait();
        for(int i = 0; i < nIters; ++i ) {
            existed = thread_id & 1;
            int oldval = locals->local(existed);
            ASSERT(existed == (i > 0), "Error on first reference");
            ASSERT(!existed || (oldval == thread_id), "Error on fetched value");
            existed = thread_id & 1;
            locals->local(existed) = thread_id;
            ASSERT(existed, "Error on assignment");
        }
    }
};

void
TestLocalAllocations( int nthread ) {
    ASSERT(nthread > 0, "nthread must be positive");
#define NITERATIONS 1000
    Body myBody(nthread, NITERATIONS);
    tbb::combinable<int> myCombinable;
    myBody.locals = &myCombinable;

    NativeParallelFor( nthread, myBody );

    int mySum = 0;
    int mySlots = 0;
    CombineEachHelperCnt<int> myCountCombine(mySum, mySlots);
    myCombinable.combine_each(myCountCombine);

    ASSERT(nthread == mySlots, "Incorrect number of slots");
    ASSERT(mySum == (nthread - 1) * nthread / 2, "Incorrect values in result");
}

void
RunLocalAllocationsTests() {
    REMARK("Testing local() allocations\n");
    for(int i = 1 <= MinThread ? MinThread : 1; i <= MaxThread; ++i) {
        REMARK("  Testing local() allocation with nthreads=%d...\n", i);
        for(int j = 0; j < 100; ++j) {
            TestLocalAllocations(i);
        }
        REMARK("  done\n");
    }
}

int TestMain () {
    if (MaxThread > 0) {
        RunParallelTests();
    }
    RunAssignmentAndCopyConstructorTests();
    RunMoveSemanticsForStateTrackableObjectTest();
    RunLocalAllocationsTests();
    return Harness::Done;
}

