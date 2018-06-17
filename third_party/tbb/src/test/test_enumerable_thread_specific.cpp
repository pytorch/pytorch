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

#include "tbb/enumerable_thread_specific.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"
#include "tbb/tbb_allocator.h"
#include "tbb/tbb_thread.h"
#include "tbb/atomic.h"

#include <cstring>
#include <vector>
#include <deque>
#include <list>
#include <map>
#include <utility>

#include "harness_assert.h"
#include "harness.h"
#include "harness_checktype.h"

#include "../tbbmalloc/shared_utils.h"
using rml::internal::estimatedCacheLineSize;

#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

static tbb::atomic<int> construction_counter;
static tbb::atomic<int> destruction_counter;

#if TBB_USE_DEBUG
const int REPETITIONS = 4;
const int N = 10000;
const int RANGE_MIN=1000;
#else
const int REPETITIONS = 10;
const int N = 100000;
const int RANGE_MIN=10000;
#endif
const int VALID_NUMBER_OF_KEYS = 100;
const double EXPECTED_SUM = (REPETITIONS + 1) * N;

//! A minimal class that occupies N bytes.
/** Defines default and copy constructor, and allows implicit operator&.
    Hides operator=. */
template<size_t N=tbb::internal::NFS_MaxLineSize>
class minimal: NoAssign {
private:
    int my_value;
    bool is_constructed;
    char pad[N-sizeof(int) - sizeof(bool)];
public:
    minimal() : NoAssign(), my_value(0) { ++construction_counter; is_constructed = true; }
    minimal( const minimal &m ) : NoAssign(), my_value(m.my_value) { ++construction_counter; is_constructed = true; }
    ~minimal() { ++destruction_counter; ASSERT(is_constructed, NULL); is_constructed = false; }
    void set_value( const int i ) { ASSERT(is_constructed, NULL); my_value = i; }
    int value( ) const { ASSERT(is_constructed, NULL); return my_value; }
};

static size_t AlignMask = 0;  // set to cache-line-size - 1

template<typename T>
T& check_alignment(T& t, const char *aname) {
    if( !tbb::internal::is_aligned(&t, AlignMask)) {
        REPORT_ONCE("alignment error with %s allocator (%x)\n", aname, (int)size_t(&t) & (AlignMask-1));
    }
    return t;
}

template<typename T>
const T& check_alignment(const T& t, const char *aname) {
    if( !tbb::internal::is_aligned(&t, AlignMask)) {
        REPORT_ONCE("alignment error with %s allocator (%x)\n", aname, (int)size_t(&t) & (AlignMask-1));
    }
    return t;
}

// Test constructors which throw.  If an ETS constructor throws before completion,
// the already-built objects are un-constructed.  Do not call the destructor if
// this occurs.

static tbb::atomic<int> gThrowValue;
static int targetThrowValue = 3;

class Thrower {
public:
    Thrower() {
#if TBB_USE_EXCEPTIONS
        if(++gThrowValue == targetThrowValue) {
            throw std::bad_alloc();
        }
#endif
    }
};

// MyThrower field of ThrowingConstructor will throw after a certain number of
// construction calls.  The constructor unwinder wshould unconstruct the instance
// of check_type<int> that was constructed just before.
class ThrowingConstructor {
    check_type<int> m_checktype;
    Thrower m_throwing_field;
public:
    int m_cnt;
    ThrowingConstructor() : m_checktype(), m_throwing_field() { m_cnt = 0;}
private:
};

//
// A helper class that simplifies writing the tests since minimal does not
// define = or + operators.
//

template< typename T >
struct test_helper {
   static inline void init(T &e) { e = static_cast<T>(0); }
   static inline void sum(T &e, const int addend ) { e += static_cast<T>(addend); }
   static inline void sum(T &e, const double addend ) { e += static_cast<T>(addend); }
   static inline void set(T &e, const int value ) { e = static_cast<T>(value); }
   static inline double get(const T &e ) { return static_cast<double>(e); }
};

template<size_t N>
struct test_helper<minimal<N> > {
   static inline void init(minimal<N> &sum) { sum.set_value( 0 ); }
   static inline void sum(minimal<N> &sum, const int addend ) { sum.set_value( sum.value() + addend); }
   static inline void sum(minimal<N> &sum, const double addend ) { sum.set_value( sum.value() + static_cast<int>(addend)); }
   static inline void sum(minimal<N> &sum, const minimal<N> &addend ) { sum.set_value( sum.value() + addend.value()); }
   static inline void set(minimal<N> &v, const int value ) { v.set_value( static_cast<int>(value) ); }
   static inline double get(const minimal<N> &sum ) { return static_cast<double>(sum.value()); }
};

template<>
struct test_helper<ThrowingConstructor> {
   static inline void init(ThrowingConstructor &sum) { sum.m_cnt = 0; }
   static inline void sum(ThrowingConstructor &sum, const int addend ) { sum.m_cnt += addend; }
   static inline void sum(ThrowingConstructor &sum, const double addend ) { sum.m_cnt += static_cast<int>(addend); }
   static inline void sum(ThrowingConstructor &sum, const ThrowingConstructor &addend ) { sum.m_cnt += addend.m_cnt; }
   static inline void set(ThrowingConstructor &v, const int value ) { v.m_cnt = static_cast<int>(value); }
   static inline double get(const ThrowingConstructor &sum ) { return static_cast<double>(sum.m_cnt); }
};

//! Tag class used to make certain constructors hard to invoke accidentally.
struct SecretTagType {} SecretTag;

//// functors and routines for initialization and combine

//! Counts instances of FunctorFinit
static tbb::atomic<int> FinitCounter;

template <typename T, int Value>
struct FunctorFinit {
    FunctorFinit( const FunctorFinit& ) {++FinitCounter;}
    FunctorFinit( SecretTagType ) {++FinitCounter;}
    ~FunctorFinit() {--FinitCounter;}
    T operator()() { return Value; }
};

template <int Value>
struct FunctorFinit<ThrowingConstructor,Value> {
    FunctorFinit( const FunctorFinit& ) {++FinitCounter;}
    FunctorFinit( SecretTagType ) {++FinitCounter;}
    ~FunctorFinit() {--FinitCounter;}
    ThrowingConstructor operator()() { ThrowingConstructor temp; temp.m_cnt = Value; return temp; }
};

template <size_t N, int Value>
struct FunctorFinit<minimal<N>,Value> {
    FunctorFinit( const FunctorFinit& ) {++FinitCounter;}
    FunctorFinit( SecretTagType ) {++FinitCounter;}
    ~FunctorFinit() {--FinitCounter;}
    minimal<N> operator()() {
        minimal<N> result;
        result.set_value( Value );
        return result;
    }
};

// Addition

template <typename T>
struct FunctorAddCombineRef {
    T operator()(const T& left, const T& right) const {
        return left+right;
    }
};

template <size_t N>
struct FunctorAddCombineRef<minimal<N> > {
    minimal<N> operator()(const minimal<N>& left, const minimal<N>& right) const {
        minimal<N> result;
        result.set_value( left.value() + right.value() );
        return result;
    }
};

template <>
struct FunctorAddCombineRef<ThrowingConstructor> {
    ThrowingConstructor operator()(const ThrowingConstructor& left, const ThrowingConstructor& right) const {
        ThrowingConstructor result;
        result.m_cnt = ( left.m_cnt + right.m_cnt );
        return result;
    }
};

template <typename T>
struct FunctorAddCombine {
    T operator()(T left, T right ) const {
        return FunctorAddCombineRef<T>()( left, right );
    }
};

template <typename T>
T FunctionAddByRef( const T &left, const T &right) {
    return FunctorAddCombineRef<T>()( left, right );
}

template <typename T>
T FunctionAdd( T left, T right) { return FunctionAddByRef(left,right); }

template <typename T>
class Accumulator {
public:
    Accumulator(T& _result) : my_result(_result) {}
    Accumulator& operator=(const Accumulator& other) {
        test_helper<T>::set(my_result, test_helper<T>::get(other));
        return *this;
    }
    void operator()(const T& new_bit) { test_helper<T>::sum(my_result, new_bit); }
private:
    T& my_result;
};

template <typename T>
class ClearingAccumulator {
public:
    ClearingAccumulator(T& _result) : my_result(_result) {}
    ClearingAccumulator& operator=(const ClearingAccumulator& other) {
        test_helper<T>::set(my_result, test_helper<T>::get(other));
        return *this;
    }
    void operator()(T& new_bit) {
        test_helper<T>::sum(my_result, new_bit);
        test_helper<T>::init(new_bit);
    }
    static void AssertClean(const T& thread_local_value) {
        T zero;
        test_helper<T>::init(zero);
        ASSERT(test_helper<T>::get(thread_local_value)==test_helper<T>::get(zero),
               "combine_each does not allow to modify thread local values?");
    }
private:
    T& my_result;
};

//// end functors and routines

template< typename T >
void run_serial_scalar_tests(const char *test_name) {
    tbb::tick_count t0;
    T sum;
    test_helper<T>::init(sum);

    REMARK("Testing serial %s... ", test_name);
    for (int t = -1; t < REPETITIONS; ++t) {
        if (Verbose && t == 0) t0 = tbb::tick_count::now();
        for (int i = 0; i < N; ++i) {
            test_helper<T>::sum(sum,1);
        }
    }

    double result_value = test_helper<T>::get(sum);
    ASSERT( EXPECTED_SUM == result_value, NULL);
    REMARK("done\nserial %s, 0, %g, %g\n", test_name, result_value, ( tbb::tick_count::now() - t0).seconds());
}


template <typename T, template<class> class Allocator>
class parallel_scalar_body: NoAssign {
    typedef tbb::enumerable_thread_specific<T, Allocator<T> > ets_type;
    ets_type &sums;
    const char* allocator_name;

public:

    parallel_scalar_body ( ets_type &_sums, const char *alloc_name ) : sums(_sums), allocator_name(alloc_name) { }

    void operator()( const tbb::blocked_range<int> &r ) const {
        for (int i = r.begin(); i != r.end(); ++i)
            test_helper<T>::sum( check_alignment(sums.local(),allocator_name), 1 );
    }

};

template< typename T, template<class> class Allocator>
void run_parallel_scalar_tests_nocombine(const char *test_name, const char *allocator_name) {

    typedef tbb::enumerable_thread_specific<T, Allocator<T> > ets_type;

    Check<T> my_check;
    gThrowValue = 0;
    {
        // We assume that static_sums zero-initialized or has a default constructor that zeros it.
        static ets_type static_sums = ets_type( T() );

        T exemplar;
        test_helper<T>::init(exemplar);

        for (int p = MinThread; p <= MaxThread; ++p) {
            REMARK("Testing parallel %s with allocator %s on %d thread(s)... ", test_name, allocator_name, p);
            tbb::task_scheduler_init init(p);
            tbb::tick_count t0;

            T iterator_sum;
            test_helper<T>::init(iterator_sum);

            T finit_ets_sum;
            test_helper<T>::init(finit_ets_sum);

            T const_iterator_sum;
            test_helper<T>::init(const_iterator_sum);

            T range_sum;
            test_helper<T>::init(range_sum);

            T const_range_sum;
            test_helper<T>::init(const_range_sum);

            T cconst_sum;
            test_helper<T>::init(cconst_sum);

            T assign_sum;
            test_helper<T>::init(assign_sum);

            T cassgn_sum;
            test_helper<T>::init(cassgn_sum);
            T non_cassgn_sum;
            test_helper<T>::init(non_cassgn_sum);

            T static_sum;
            test_helper<T>::init(static_sum);

            for (int t = -1; t < REPETITIONS; ++t) {
                if (Verbose && t == 0) t0 = tbb::tick_count::now();

                static_sums.clear();

                ets_type sums(exemplar);
                FunctorFinit<T,0> my_finit(SecretTag);
                ets_type finit_ets(my_finit);

                ASSERT( sums.empty(), NULL);
                tbb::parallel_for( tbb::blocked_range<int>( 0, N, RANGE_MIN ), parallel_scalar_body<T,Allocator>( sums, allocator_name ) );
                ASSERT( !sums.empty(), NULL);

                ASSERT( finit_ets.empty(), NULL);
                tbb::parallel_for( tbb::blocked_range<int>( 0, N, RANGE_MIN ), parallel_scalar_body<T,Allocator>( finit_ets, allocator_name ) );
                ASSERT( !finit_ets.empty(), NULL);

                ASSERT(static_sums.empty(), NULL);
                tbb::parallel_for( tbb::blocked_range<int>( 0, N, RANGE_MIN ), parallel_scalar_body<T,Allocator>( static_sums, allocator_name ) );
                ASSERT( !static_sums.empty(), NULL);

                // use iterator
                typename ets_type::size_type size = 0;
                for ( typename ets_type::iterator i = sums.begin(); i != sums.end(); ++i ) {
                     ++size;
                     test_helper<T>::sum(iterator_sum, *i);
                }
                ASSERT( sums.size() == size, NULL);

                // use const_iterator
                for ( typename ets_type::const_iterator i = sums.begin(); i != sums.end(); ++i ) {
                     test_helper<T>::sum(const_iterator_sum, *i);
                }

                // use range_type
                typename ets_type::range_type r = sums.range();
                for ( typename ets_type::range_type::const_iterator i = r.begin(); i != r.end(); ++i ) {
                     test_helper<T>::sum(range_sum, *i);
                }

                // use const_range_type
                typename ets_type::const_range_type cr = sums.range();
                for ( typename ets_type::const_range_type::iterator i = cr.begin(); i != cr.end(); ++i ) {
                     test_helper<T>::sum(const_range_sum, *i);
                }

                // test copy constructor, with TLS-cached locals
                typedef typename tbb::enumerable_thread_specific<T, Allocator<T>, tbb::ets_key_per_instance> cached_ets_type;

                cached_ets_type cconst(sums);

                for ( typename cached_ets_type::const_iterator i = cconst.begin(); i != cconst.end(); ++i ) {
                     test_helper<T>::sum(cconst_sum, *i);
                }

                // test assignment
                ets_type assigned;
                assigned = sums;

                for ( typename ets_type::const_iterator i = assigned.begin(); i != assigned.end(); ++i ) {
                     test_helper<T>::sum(assign_sum, *i);
                }

                // test assign to and from cached locals
                cached_ets_type cassgn;
                cassgn = sums;
                for ( typename cached_ets_type::const_iterator i = cassgn.begin(); i != cassgn.end(); ++i ) {
                     test_helper<T>::sum(cassgn_sum, *i);
                }

                ets_type non_cassgn;
                non_cassgn = cassgn;
                for ( typename ets_type::const_iterator i = non_cassgn.begin(); i != non_cassgn.end(); ++i ) {
                     test_helper<T>::sum(non_cassgn_sum, *i);
                }

                // test finit-initialized ets
                for(typename ets_type::const_iterator i = finit_ets.begin(); i != finit_ets.end(); ++i) {
                    test_helper<T>::sum(finit_ets_sum, *i);
                }

                // test static ets
                for(typename ets_type::const_iterator i = static_sums.begin(); i != static_sums.end(); ++i) {
                    test_helper<T>::sum(static_sum, *i);
                }

            }

            ASSERT( EXPECTED_SUM == test_helper<T>::get(iterator_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(const_iterator_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(range_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(const_range_sum), NULL);

            ASSERT( EXPECTED_SUM == test_helper<T>::get(cconst_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(assign_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(cassgn_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(non_cassgn_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(finit_ets_sum), NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(static_sum), NULL);

            REMARK("done\nparallel %s, %d, %g, %g\n", test_name, p, test_helper<T>::get(iterator_sum),
                                                          ( tbb::tick_count::now() - t0).seconds());
        }
    }  // Check block
}

template< typename T, template<class> class Allocator>
void run_parallel_scalar_tests(const char *test_name, const char *allocator_name) {

    typedef tbb::enumerable_thread_specific<T, Allocator<T> > ets_type;
    bool exception_caught = false;

    // We assume that static_sums zero-initialized or has a default constructor that zeros it.
    static ets_type static_sums = ets_type( T() );

    T exemplar;
    test_helper<T>::init(exemplar);

    int test_throw_count = 10;
    // the test will be performed repeatedly until it does not throw.  For non-throwing types
    // this means once; for the throwing type test it may loop two or three times.  The
    // value of targetThrowValue will determine when and if the test will throw.
    do {
        targetThrowValue = test_throw_count;  // keep testing until we get no exception
        exception_caught = false;
#if TBB_USE_EXCEPTIONS
        try {
#endif
            run_parallel_scalar_tests_nocombine<T,Allocator>(test_name, allocator_name);
#if TBB_USE_EXCEPTIONS
        }
        catch(...) {
            REMARK("Exception caught %d\n", targetThrowValue);
        }
#endif
        for (int p = MinThread; p <= MaxThread; ++p) {
            REMARK("Testing parallel %s with allocator %s on %d thread(s)... ", test_name, allocator_name, p);
            tbb::task_scheduler_init init(p);
            tbb::tick_count t0;

            gThrowValue = 0;

            T combine_sum;
            test_helper<T>::init(combine_sum);

            T combine_ref_sum;
            test_helper<T>::init(combine_ref_sum);

            T accumulator_sum;
            test_helper<T>::init(accumulator_sum);

            T static_sum;
            test_helper<T>::init(static_sum);

            T clearing_accumulator_sum;
            test_helper<T>::init(clearing_accumulator_sum);

            {
                Check<T> my_check;
#if TBB_USE_EXCEPTIONS
                try
#endif
                {
                    for (int t = -1; t < REPETITIONS; ++t) {
                        if (Verbose && t == 0) t0 = tbb::tick_count::now();

                        static_sums.clear();

                        ets_type sums(exemplar);

                        ASSERT( sums.empty(), NULL);
                        tbb::parallel_for( tbb::blocked_range<int>( 0, N, RANGE_MIN ),
                                parallel_scalar_body<T,Allocator>( sums, allocator_name ) );
                        ASSERT( !sums.empty(), NULL);

                        ASSERT(static_sums.empty(), NULL);
                        tbb::parallel_for( tbb::blocked_range<int>( 0, N, RANGE_MIN ),
                                parallel_scalar_body<T,Allocator>( static_sums, allocator_name ) );
                        ASSERT( !static_sums.empty(), NULL);

                        // Use combine
                        test_helper<T>::sum(combine_sum, sums.combine(FunctionAdd<T>));
                        test_helper<T>::sum(combine_ref_sum, sums.combine(FunctionAddByRef<T>));
                        test_helper<T>::sum(static_sum, static_sums.combine(FunctionAdd<T>));

                        // Accumulate with combine_each
                        sums.combine_each(Accumulator<T>(accumulator_sum));
                        // Accumulate and clear thread-local values
                        sums.combine_each(ClearingAccumulator<T>(clearing_accumulator_sum));
                        // Check that the values were cleared
                        sums.combine_each(ClearingAccumulator<T>::AssertClean);
                    }
                }
#if TBB_USE_EXCEPTIONS
                catch(...) {
                    REMARK("Exception caught %d\n", targetThrowValue);
                    exception_caught = true;
                }
#endif
            }

            ASSERT( EXPECTED_SUM == test_helper<T>::get(combine_sum) || exception_caught, NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(combine_ref_sum) || exception_caught, NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(static_sum) || exception_caught, NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(accumulator_sum) || exception_caught, NULL);
            ASSERT( EXPECTED_SUM == test_helper<T>::get(clearing_accumulator_sum) || exception_caught, NULL);

            REMARK("done\nparallel combine %s, %d, %g, %g\n", test_name, p, test_helper<T>::get(combine_sum),
                                                          ( tbb::tick_count::now() - t0).seconds());
        }  // MinThread .. MaxThread
        test_throw_count += 10;  // keep testing until we don't get an exception
    } while (exception_caught && test_throw_count < 200);
    ASSERT(!exception_caught, "No non-exception test completed");
}

template <typename T, template<class> class Allocator>
class parallel_vector_for_body: NoAssign {
    typedef std::vector<T, tbb::tbb_allocator<T> > container_type;
    typedef tbb::enumerable_thread_specific< container_type, Allocator<container_type> > ets_type;
    ets_type &locals;
    const char *allocator_name;

public:

    parallel_vector_for_body ( ets_type &_locals, const char *aname ) : locals(_locals), allocator_name(aname) { }

    void operator()( const tbb::blocked_range<int> &r ) const {
        T one;
        test_helper<T>::set(one, 1);

        for (int i = r.begin(); i < r.end(); ++i) {
            check_alignment(locals.local(),allocator_name).push_back( one );
        }
    }

};

template <typename R, typename T>
struct parallel_vector_reduce_body {

    T sum;
    size_t count;
    typedef std::vector<T, tbb::tbb_allocator<T> > container_type;

    parallel_vector_reduce_body ( ) : count(0) { test_helper<T>::init(sum); }
    parallel_vector_reduce_body ( parallel_vector_reduce_body<R, T> &, tbb::split ) : count(0) {  test_helper<T>::init(sum); }

    void operator()( const R &r ) {
        for (typename R::iterator ri = r.begin(); ri != r.end(); ++ri) {
            const container_type &v = *ri;
            ++count;
            for (typename container_type::const_iterator vi = v.begin(); vi != v.end(); ++vi) {
                test_helper<T>::sum(sum, *vi);
            }
        }
    }

    void join( const parallel_vector_reduce_body &b ) {
        test_helper<T>::sum(sum,b.sum);
        count += b.count;
    }

};

template< typename T, template<class> class Allocator>
void run_parallel_vector_tests(const char *test_name, const char *allocator_name) {
    tbb::tick_count t0;
    typedef std::vector<T, tbb::tbb_allocator<T> > container_type;
    typedef tbb::enumerable_thread_specific< container_type, Allocator<container_type> > ets_type;

    for (int p = MinThread; p <= MaxThread; ++p) {
        REMARK("Testing parallel %s with allocator %s on %d thread(s)... ", test_name, allocator_name, p);
        tbb::task_scheduler_init init(p);

        T sum;
        test_helper<T>::init(sum);

        for (int t = -1; t < REPETITIONS; ++t) {
            if (Verbose && t == 0) t0 = tbb::tick_count::now();
            ets_type vs;

            ASSERT( vs.empty(), NULL );
            tbb::parallel_for( tbb::blocked_range<int> (0, N, RANGE_MIN),
                               parallel_vector_for_body<T,Allocator>( vs, allocator_name ) );
            ASSERT( !vs.empty(), NULL );

            // copy construct
            ets_type vs2(vs); // this causes an assertion failure, related to allocators...

            // assign
            ets_type vs3;
            vs3 = vs;

            parallel_vector_reduce_body< typename ets_type::const_range_type, T > pvrb;
            tbb::parallel_reduce ( vs.range(1), pvrb );

            test_helper<T>::sum(sum, pvrb.sum);

            ASSERT( vs.size() == pvrb.count, NULL );
            ASSERT( vs2.size() == pvrb.count, NULL );
            ASSERT( vs3.size() == pvrb.count, NULL );

            tbb::flattened2d<ets_type> fvs = flatten2d(vs);
            size_t ccount = fvs.size();
            ASSERT( ccount == size_t(N), NULL );
            size_t elem_cnt = 0;
            for(typename tbb::flattened2d<ets_type>::const_iterator i = fvs.begin(); i != fvs.end(); ++i) {
                ++elem_cnt;
            };
            ASSERT( ccount == elem_cnt, NULL );

            elem_cnt = 0;
            for(typename tbb::flattened2d<ets_type>::iterator i = fvs.begin(); i != fvs.end(); ++i) {
                ++elem_cnt;
            };
            ASSERT( ccount == elem_cnt, NULL );

#if __TBB_ETS_USE_CPP11
            // Test the ETS constructor with multiple args
            T minus_one;
            test_helper<T>::set(minus_one, -1);
            // Set ETS to construct "local" vectors pre-occupied with 25 "minus_one"s
            // Cast 25 to size_type to prevent Intel Compiler SFINAE compilation issues with gcc 5.
            ets_type vvs( typename container_type::size_type(25), minus_one, tbb::tbb_allocator<T>() );
            ASSERT( vvs.empty(), NULL );
            tbb::parallel_for ( tbb::blocked_range<int> (0, N, RANGE_MIN), parallel_vector_for_body<T,Allocator>( vvs, allocator_name ) );
            ASSERT( !vvs.empty(), NULL );

            parallel_vector_reduce_body< typename ets_type::const_range_type, T > pvrb2;
            tbb::parallel_reduce ( vvs.range(1), pvrb2 );
            ASSERT( pvrb2.count == vvs.size(), NULL );
            ASSERT( test_helper<T>::get(pvrb2.sum) == N-pvrb2.count*25, NULL );

            tbb::flattened2d<ets_type> fvvs = flatten2d(vvs);
            ccount = fvvs.size();
            ASSERT( ccount == N+pvrb2.count*25, NULL );
#endif
        }

        double result_value = test_helper<T>::get(sum);
        ASSERT( EXPECTED_SUM == result_value, NULL);
        REMARK("done\nparallel %s, %d, %g, %g\n", test_name, p, result_value, ( tbb::tick_count::now() - t0).seconds());
    }
}

template<typename T, template<class> class Allocator>
void run_cross_type_vector_tests(const char *test_name) {
    tbb::tick_count t0;
    const char* allocator_name = "default";
    typedef std::vector<T, tbb::tbb_allocator<T> > container_type;

    for (int p = MinThread; p <= MaxThread; ++p) {
        REMARK("Testing parallel %s on %d thread(s)... ", test_name, p);
        tbb::task_scheduler_init init(p);

        T sum;
        test_helper<T>::init(sum);

        for (int t = -1; t < REPETITIONS; ++t) {
            if (Verbose && t == 0) t0 = tbb::tick_count::now();
            typedef typename tbb::enumerable_thread_specific< container_type, Allocator<container_type>, tbb::ets_no_key > ets_nokey_type;
            typedef typename tbb::enumerable_thread_specific< container_type, Allocator<container_type>, tbb::ets_key_per_instance > ets_tlskey_type;
            ets_nokey_type vs;

            ASSERT( vs.empty(), NULL);
            tbb::parallel_for ( tbb::blocked_range<int> (0, N, RANGE_MIN), parallel_vector_for_body<T, Allocator>( vs, allocator_name ) );
            ASSERT( !vs.empty(), NULL);

            // copy construct
            ets_tlskey_type vs2(vs);

            // assign
            ets_nokey_type vs3;
            vs3 = vs2;

            parallel_vector_reduce_body< typename ets_nokey_type::const_range_type, T > pvrb;
            tbb::parallel_reduce ( vs3.range(1), pvrb );

            test_helper<T>::sum(sum, pvrb.sum);

            ASSERT( vs3.size() == pvrb.count, NULL);

            tbb::flattened2d<ets_nokey_type> fvs = flatten2d(vs3);
            size_t ccount = fvs.size();
            size_t elem_cnt = 0;
            for(typename tbb::flattened2d<ets_nokey_type>::const_iterator i = fvs.begin(); i != fvs.end(); ++i) {
                ++elem_cnt;
            };
            ASSERT(ccount == elem_cnt, NULL);

            elem_cnt = 0;
            for(typename tbb::flattened2d<ets_nokey_type>::iterator i = fvs.begin(); i != fvs.end(); ++i) {
                ++elem_cnt;
            };
            ASSERT(ccount == elem_cnt, NULL);
        }

        double result_value = test_helper<T>::get(sum);
        ASSERT( EXPECTED_SUM == result_value, NULL);
        REMARK("done\nparallel %s, %d, %g, %g\n", test_name, p, result_value, ( tbb::tick_count::now() - t0).seconds());
    }
}

template< typename T >
void run_serial_vector_tests(const char *test_name) {
    tbb::tick_count t0;
    T sum;
    test_helper<T>::init(sum);
    T one;
    test_helper<T>::set(one, 1);

    REMARK("Testing serial %s... ", test_name);
    for (int t = -1; t < REPETITIONS; ++t) {
        if (Verbose && t == 0) t0 = tbb::tick_count::now();
        std::vector<T, tbb::tbb_allocator<T> > v;
        for (int i = 0; i < N; ++i) {
            v.push_back( one );
        }
        for (typename std::vector<T, tbb::tbb_allocator<T> >::const_iterator i = v.begin(); i != v.end(); ++i)
            test_helper<T>::sum(sum, *i);
    }

    double result_value = test_helper<T>::get(sum);
    ASSERT( EXPECTED_SUM == result_value, NULL);
    REMARK("done\nserial %s, 0, %g, %g\n", test_name, result_value, ( tbb::tick_count::now() - t0).seconds());
}

const size_t line_size = tbb::internal::NFS_MaxLineSize;

void run_serial_tests() {
    run_serial_scalar_tests<int>("int");
    run_serial_scalar_tests<double>("double");
    run_serial_scalar_tests<minimal<> >("minimal<>");
    run_serial_vector_tests<int>("std::vector<int, tbb::tbb_allocator<int> >");
    run_serial_vector_tests<double>("std::vector<double, tbb::tbb_allocator<double> >");
}

template<template<class>class Allocator>
void run_parallel_tests(const char *allocator_name) {
    run_parallel_scalar_tests<int, Allocator>("int",allocator_name);
    run_parallel_scalar_tests<double, Allocator>("double",allocator_name);
    run_parallel_scalar_tests_nocombine<minimal<>,Allocator>("minimal<>",allocator_name);
    run_parallel_scalar_tests<ThrowingConstructor, Allocator>("ThrowingConstructor", allocator_name);
    run_parallel_vector_tests<int, Allocator>("std::vector<int, tbb::tbb_allocator<int> >",allocator_name);
    run_parallel_vector_tests<double, Allocator>("std::vector<double, tbb::tbb_allocator<double> >",allocator_name);
}

void run_cross_type_tests() {
    // cross-type scalar tests are part of run_parallel_scalar_tests_nocombine
    run_cross_type_vector_tests<int, tbb::tbb_allocator>("std::vector<int, tbb::tbb_allocator<int> >");
    run_cross_type_vector_tests<double, tbb::tbb_allocator>("std::vector<double, tbb::tbb_allocator<double> >");
}

typedef tbb::enumerable_thread_specific<minimal<line_size> > flogged_ets;

class set_body {
    flogged_ets *a;

public:
    set_body( flogged_ets*_a ) : a(_a) { }

    void operator() ( ) const {
        for (int i = 0; i < VALID_NUMBER_OF_KEYS; ++i) {
            check_alignment(a[i].local(), "default").set_value(i + 1);
        }
    }

};

void do_tbb_threads( int max_threads, flogged_ets a[] ) {
    std::vector< tbb::tbb_thread * > threads;

    for (int p = 0; p < max_threads; ++p) {
        threads.push_back( new tbb::tbb_thread ( set_body( a ) ) );
    }

    for (int p = 0; p < max_threads; ++p) {
        threads[p]->join();
    }

    for(int p = 0; p < max_threads; ++p) {
        delete threads[p];
    }
}

void flog_key_creation_and_deletion() {
    const int FLOG_REPETITIONS = 100;

    for (int p = MinThread; p <= MaxThread; ++p) {
        REMARK("Testing repeated deletes on %d threads... ", p);

        for (int j = 0; j < FLOG_REPETITIONS; ++j) {
            construction_counter = 0;
            destruction_counter = 0;

            // causes VALID_NUMBER_OF_KEYS exemplar instances to be constructed
            flogged_ets* a = new flogged_ets[VALID_NUMBER_OF_KEYS];
            ASSERT(int(construction_counter) == 0, NULL);   // no exemplars or actual locals have been constructed
            ASSERT(int(destruction_counter) == 0, NULL);    // and none have been destroyed

            // causes p * VALID_NUMBER_OF_KEYS minimals to be created
            do_tbb_threads(p, a);

            for (int i = 0; i < VALID_NUMBER_OF_KEYS; ++i) {
                int pcnt = 0;
                for ( flogged_ets::iterator tli = a[i].begin(); tli != a[i].end(); ++tli ) {
                    ASSERT( (*tli).value() == i+1, NULL );
                    ++pcnt;
                }
                ASSERT( pcnt == p, NULL);  // should be one local per thread.
            }
            delete[] a;
        }

        ASSERT( int(construction_counter) == (p)*VALID_NUMBER_OF_KEYS, NULL );
        ASSERT( int(destruction_counter) == (p)*VALID_NUMBER_OF_KEYS, NULL );

        REMARK("done\nTesting repeated clears on %d threads... ", p);

        construction_counter = 0;
        destruction_counter = 0;

        // causes VALID_NUMBER_OF_KEYS exemplar instances to be constructed
        flogged_ets* a = new flogged_ets[VALID_NUMBER_OF_KEYS];

        for (int j = 0; j < FLOG_REPETITIONS; ++j) {

            // causes p * VALID_NUMBER_OF_KEYS minimals to be created
            do_tbb_threads(p, a);

            for (int i = 0; i < VALID_NUMBER_OF_KEYS; ++i) {
                for ( flogged_ets::iterator tli = a[i].begin(); tli != a[i].end(); ++tli ) {
                    ASSERT( (*tli).value() == i+1, NULL );
                }
                a[i].clear();
                ASSERT( static_cast<int>(a[i].end() - a[i].begin()) == 0, NULL );
            }

        }

        delete[] a;

        ASSERT( int(construction_counter) == (FLOG_REPETITIONS*p)*VALID_NUMBER_OF_KEYS, NULL );
        ASSERT( int(destruction_counter) == (FLOG_REPETITIONS*p)*VALID_NUMBER_OF_KEYS, NULL );

        REMARK("done\n");
    }

}

template <typename inner_container>
void flog_segmented_interator() {

    bool found_error = false;
    typedef typename inner_container::value_type T;
    typedef std::vector< inner_container > nested_vec;
    inner_container my_inner_container;
    my_inner_container.clear();
    nested_vec my_vec;

    // simple nested vector (neither level empty)
    const int maxval = 10;
    for(int i=0; i < maxval; i++) {
        my_vec.push_back(my_inner_container);
        for(int j = 0; j < maxval; j++) {
            my_vec.at(i).push_back((T)(maxval * i + j));
        }
    }

    tbb::internal::segmented_iterator<nested_vec, T> my_si(my_vec);

    T ii;
    for(my_si=my_vec.begin(), ii=0; my_si != my_vec.end(); ++my_si, ++ii) {
        if((*my_si) != ii) {
            found_error = true;
            REMARK( "*my_si=%d\n", int(*my_si));
        }
    }

    // outer level empty
    my_vec.clear();
    for(my_si=my_vec.begin(); my_si != my_vec.end(); ++my_si) {
        found_error = true;
    }

    // inner levels empty
    my_vec.clear();
    for(int i =0; i < maxval; ++i) {
        my_vec.push_back(my_inner_container);
    }
    for(my_si = my_vec.begin(); my_si != my_vec.end(); ++my_si) {
        found_error = true;
    }

    // every other inner container is empty
    my_vec.clear();
    for(int i=0; i < maxval; ++i) {
        my_vec.push_back(my_inner_container);
        if(i%2) {
            for(int j = 0; j < maxval; ++j) {
                my_vec.at(i).push_back((T)(maxval * (i/2) + j));
            }
        }
    }
    for(my_si = my_vec.begin(), ii=0; my_si != my_vec.end(); ++my_si, ++ii) {
        if((*my_si) != ii) {
            found_error = true;
            REMARK("*my_si=%d, ii=%d\n", (int)(*my_si), (int)ii);
        }
    }

    tbb::internal::segmented_iterator<nested_vec, const T> my_csi(my_vec);
    for(my_csi=my_vec.begin(), ii=0; my_csi != my_vec.end(); ++my_csi, ++ii) {
        if((*my_csi) != ii) {
            found_error = true;
            REMARK( "*my_csi=%d\n", int(*my_csi));
        }
    }

    // outer level empty
    my_vec.clear();
    for(my_csi=my_vec.begin(); my_csi != my_vec.end(); ++my_csi) {
        found_error = true;
    }

    // inner levels empty
    my_vec.clear();
    for(int i =0; i < maxval; ++i) {
        my_vec.push_back(my_inner_container);
    }
    for(my_csi = my_vec.begin(); my_csi != my_vec.end(); ++my_csi) {
        found_error = true;
    }

    // every other inner container is empty
    my_vec.clear();
    for(int i=0; i < maxval; ++i) {
        my_vec.push_back(my_inner_container);
        if(i%2) {
            for(int j = 0; j < maxval; ++j) {
                my_vec.at(i).push_back((T)(maxval * (i/2) + j));
            }
        }
    }
    for(my_csi = my_vec.begin(), ii=0; my_csi != my_vec.end(); ++my_csi, ++ii) {
        if((*my_csi) != ii) {
            found_error = true;
            REMARK("*my_csi=%d, ii=%d\n", (int)(*my_csi), (int)ii);
        }
    }


    if(found_error) REPORT("segmented_iterator failed\n");
}

template <typename Key, typename Val>
void flog_segmented_iterator_map() {
   typedef typename std::map<Key, Val> my_map;
   typedef std::vector< my_map > nested_vec;
   my_map my_inner_container;
   my_inner_container.clear();
   nested_vec my_vec;
   my_vec.clear();
   bool found_error = false;

   // simple nested vector (neither level empty)
   const int maxval = 4;
   for(int i=0; i < maxval; i++) {
       my_vec.push_back(my_inner_container);
       for(int j = 0; j < maxval; j++) {
           my_vec.at(i).insert(std::make_pair<Key,Val>(maxval * i + j, 2*(maxval*i + j)));
       }
   }

   tbb::internal::segmented_iterator<nested_vec, std::pair<const Key, Val> > my_si(my_vec);
   Key ii;
   for(my_si=my_vec.begin(), ii=0; my_si != my_vec.end(); ++my_si, ++ii) {
       if(((*my_si).first != ii) || ((*my_si).second != 2*ii)) {
           found_error = true;
           REMARK( "ii=%d, (*my_si).first=%d, second=%d\n",ii, int((*my_si).first), int((*my_si).second));
       }
   }

   tbb::internal::segmented_iterator<nested_vec, const std::pair<const Key, Val> > my_csi(my_vec);
   for(my_csi=my_vec.begin(), ii=0; my_csi != my_vec.end(); ++my_csi, ++ii) {
       if(((*my_csi).first != ii) || ((*my_csi).second != 2*ii)) {
           found_error = true;
           REMARK( "ii=%d, (*my_csi).first=%d, second=%d\n",ii, int((*my_csi).first), int((*my_csi).second));
       }
   }
   if(found_error) REPORT("segmented_iterator_map failed\n");
}

void run_segmented_iterator_tests() {
   // only the following containers can be used with the segmented iterator.
   REMARK("Running Segmented Iterator Tests\n");
   flog_segmented_interator<std::vector< int > >();
   flog_segmented_interator<std::vector< double > >();
   flog_segmented_interator<std::deque< int > >();
   flog_segmented_interator<std::deque< double > >();
   flog_segmented_interator<std::list< int > >();
   flog_segmented_interator<std::list< double > >();

   flog_segmented_iterator_map<int, int>();
   flog_segmented_iterator_map<int, double>();
}

template<typename T, template<class> class Allocator, typename Init>
tbb::enumerable_thread_specific<T,Allocator<T> > MakeETS( Init init ) {
    return tbb::enumerable_thread_specific<T,Allocator<T> >(init);
}
#if __TBB_ETS_USE_CPP11
// In some GCC versions, parameter packs in lambdas might cause compile errors
template<typename ETS, typename... P>
struct MakeETS_Functor {
    ETS operator()( typename tbb::internal::strip<P>::type&&... params ) {
        return ETS(std::move(params)...);
    }
};
template<typename T, template<class> class Allocator, typename... P>
tbb::enumerable_thread_specific<T,Allocator<T> > MakeETS( tbb::internal::stored_pack<P...> pack ) {
    typedef tbb::enumerable_thread_specific<T,Allocator<T> > result_type;
    return tbb::internal::call_and_return< result_type >(
        MakeETS_Functor<result_type,P...>(), std::move(pack)
    );
}
#endif

template<typename T, template<class> class Allocator, typename InitSrc, typename InitDst, typename Validator>
void ets_copy_assign_test( InitSrc init1, InitDst init2, Validator check, const char *allocator_name ) {
    typedef tbb::enumerable_thread_specific<T, Allocator<T> > ets_type;

    // Create the source instance
    const ets_type& cref_binder = MakeETS<T, Allocator>(init1);
    ets_type& source = const_cast<ets_type&>(cref_binder);
    check(check_alignment(source.local(),allocator_name));

    // Test copy construction
    bool existed = false;
    ets_type copy(source);
    check(check_alignment(copy.local(existed),allocator_name));
    ASSERT(existed, "Local data not created by ETS copy constructor");
    copy.clear();
    check(check_alignment(copy.local(),allocator_name));

    // Test assignment
    existed = false;
    ets_type assign(init2);
    assign = source;
    check(check_alignment(assign.local(existed),allocator_name));
    ASSERT(existed, "Local data not created by ETS assignment");
    assign.clear();
    check(check_alignment(assign.local(),allocator_name));

#if __TBB_ETS_USE_CPP11
    // Create the source instance
    ets_type&& rvref_binder = MakeETS<T, Allocator>(init1);
    check(check_alignment(rvref_binder.local(),allocator_name));

    // Test move construction
    existed = false;
    ets_type moved(rvref_binder);
    check(check_alignment(moved.local(existed),allocator_name));
    ASSERT(existed, "Local data not created by ETS move constructor");
    moved.clear();
    check(check_alignment(moved.local(),allocator_name));

    // Test assignment
    existed = false;
    ets_type move_assign(init2);
    move_assign = std::move(moved);
    check(check_alignment(move_assign.local(existed),allocator_name));
    ASSERT(existed, "Local data not created by ETS move assignment");
    move_assign.clear();
    check(check_alignment(move_assign.local(),allocator_name));
#endif
}

template<typename T, int Expected>
struct Validator {
    void operator()( const T& value ) {
        ASSERT(test_helper<T>::get(value) == Expected, NULL);
    }
    void operator()( const std::pair<int,T>& value ) {
        ASSERT(value.first > 0, NULL);
        ASSERT(test_helper<T>::get(value.second) == Expected*value.first, NULL);
    }
};

template <typename T, template<class> class Allocator>
void run_assign_and_copy_constructor_test(const char *test_name, const char *allocator_name) {
    REMARK("Testing assignment and copy construction for %s with allocator %s\n", test_name, allocator_name);
    #define EXPECTED 3142

    // test with exemplar initializer
    T src_init;
    test_helper<T>::set(src_init,EXPECTED);
    T other_init;
    test_helper<T>::init(other_init);
    ets_copy_assign_test<T, Allocator>(src_init, other_init, Validator<T,EXPECTED>(), allocator_name);

    // test with function initializer
    FunctorFinit<T,EXPECTED> src_finit(SecretTag);
    FunctorFinit<T,0> other_finit(SecretTag);
    ets_copy_assign_test<T, Allocator>(src_finit, other_finit, Validator<T,EXPECTED>(), allocator_name);

#if __TBB_ETS_USE_CPP11
    // test with multi-argument "emplace" initializer
    // The arguments are wrapped into tbb::internal::stored_pack to avoid variadic templates in ets_copy_assign_test.
    test_helper<T>::set(src_init,EXPECTED*17);
    ets_copy_assign_test< std::pair<int,T>, Allocator>(tbb::internal::save_pack(17,src_init), std::make_pair(-1,T()), Validator<T,EXPECTED>(), allocator_name);
#endif
    #undef EXPECTED
}

template< template<class> class Allocator>
void run_assignment_and_copy_constructor_tests(const char* allocator_name) {
    REMARK("Running assignment and copy constructor tests\n");
    run_assign_and_copy_constructor_test<int, Allocator>("int", allocator_name);
    run_assign_and_copy_constructor_test<double, Allocator>("double", allocator_name);
    // Try class sizes that are close to a cache line in size, in order to check padding calculations.
    run_assign_and_copy_constructor_test<minimal<line_size-1>, Allocator >("minimal<line_size-1>", allocator_name);
    run_assign_and_copy_constructor_test<minimal<line_size>, Allocator >("minimal<line_size>", allocator_name);
    run_assign_and_copy_constructor_test<minimal<line_size+1>, Allocator >("minimal<line_size+1>", allocator_name);
    ASSERT(FinitCounter==0, NULL);
}

// Class with no default constructor
class HasNoDefaultConstructor {
    HasNoDefaultConstructor();
public:
    HasNoDefaultConstructor( SecretTagType ) {}
};
// Initialization functor for HasNoDefaultConstructor
struct HasNoDefaultConstructorFinit {
    HasNoDefaultConstructor operator()() {
        return HasNoDefaultConstructor(SecretTag);
    }
};
// Combine functor for HasNoDefaultConstructor
struct HasNoDefaultConstructorCombine {
    HasNoDefaultConstructor operator()( HasNoDefaultConstructor, HasNoDefaultConstructor ) {
        return HasNoDefaultConstructor(SecretTag);
    }
};

#if __TBB_ETS_USE_CPP11
// Class that only has a constructor with multiple parameters and a move constructor
class HasSpecialAndMoveCtor : NoCopy {
    HasSpecialAndMoveCtor();
public:
    HasSpecialAndMoveCtor( SecretTagType, size_t = size_t(0), const char* = "" ) {}
    HasSpecialAndMoveCtor( HasSpecialAndMoveCtor&& ) {}
};
#endif

// No-op combine-each functor
template<typename V>
struct EmptyCombineEach {
    void operator()( const V& ) { }
};

int
align_val(void * const p) {
    size_t tmp = (size_t)p;
    int a = 1;
    while((tmp&0x1) == 0) { a <<=1; tmp >>= 1; }
    return a;
}

bool is_between(void* lowp, void *highp, void *testp) {
    if((size_t)lowp < (size_t)testp && (size_t)testp < (size_t)highp) return true;
    return (size_t)lowp > (size_t)testp && (size_t)testp > (size_t)highp;
}

template<class U> struct alignment_of {
    typedef struct { char t; U    padded; } test_alignment;
    static const size_t value = sizeof(test_alignment) - sizeof(U);
};
using tbb::interface6::internal::ets_element;
template<typename T, typename OtherType>
void allocate_ets_element_on_stack(const char *name) {
    typedef T aligning_element_type;
    const size_t my_align = alignment_of<aligning_element_type>::value;
    OtherType c1;
    ets_element<aligning_element_type> my_stack_element;
    OtherType c2;
    ets_element<aligning_element_type> my_stack_element2;
    struct {
        OtherType cxx;
        ets_element<aligning_element_type> my_struct_element;
    } mystruct1;
    tbb::internal::suppress_unused_warning(c1,c2);
    REMARK("using %s, c1 address == %lx (alignment %d), c2 address == %lx (alignment %d)\n", name, &c1, align_val(&c1), &c2, align_val(&c2));
    REMARK(" ---- my_align == %d\n", (int)my_align);
    REMARK("    my_stack_element == %lx (alignment %d), my_stack_element2 == %lx (alignment %d)\n",
            &my_stack_element, align_val(&my_stack_element), &my_stack_element2, align_val(&my_stack_element2));
    if(is_between(&c1,&c2,&my_stack_element)) REMARK("my_struct_element is in the middle\n");
    if(is_between(&c1,&c2,&my_stack_element2)) REMARK("my_struct_element2 is in the middle\n");
    if(!is_between(&c1,&c2,&my_stack_element) && !is_between(&c1,&c2,&my_stack_element2)) REMARK("stack vars reorganized\n");
    REMARK("   structure field address == %lx, alignment %d\n",
            mystruct1.my_struct_element.value(),
            align_val(mystruct1.my_struct_element.value())
            );
    ASSERT(tbb::internal::is_aligned(my_stack_element.value(), my_align), "Error in first stack alignment" );
    ASSERT(tbb::internal::is_aligned(my_stack_element2.value(), my_align), "Error in second stack alignment" );
    ASSERT(tbb::internal::is_aligned(mystruct1.my_struct_element.value(), my_align), "Error in struct element alignment" );
}

//! Test situations where only default constructor or copy constructor is required.
template<template<class> class Allocator>
void TestInstantiation(const char *allocator_name) {
    REMARK("TestInstantiation<%s>\n", allocator_name);
    // Test instantiation is possible when copy constructor is not required.
    tbb::enumerable_thread_specific<NoCopy, Allocator<NoCopy> > ets1;
    ets1.local();
    ets1.combine_each(EmptyCombineEach<NoCopy>());

    // Test instantiation when default constructor is not required, because exemplar is provided.
    HasNoDefaultConstructor x(SecretTag);
    tbb::enumerable_thread_specific<HasNoDefaultConstructor, Allocator<HasNoDefaultConstructor> > ets2(x);
    ets2.local();
    ets2.combine(HasNoDefaultConstructorCombine());

    // Test instantiation when default constructor is not required, because init function is provided.
    HasNoDefaultConstructorFinit f;
    tbb::enumerable_thread_specific<HasNoDefaultConstructor, Allocator<HasNoDefaultConstructor> > ets3(f);
    ets3.local();
    ets3.combine(HasNoDefaultConstructorCombine());

#if __TBB_ETS_USE_CPP11
    // Test instantiation with multiple arguments
    tbb::enumerable_thread_specific<HasSpecialAndMoveCtor, Allocator<HasSpecialAndMoveCtor> > ets4(SecretTag, 0x42, "meaningless");
    ets4.local();
    ets4.combine_each(EmptyCombineEach<HasSpecialAndMoveCtor>());
    // Test instantiation with one argument that should however use the variadic constructor
    tbb::enumerable_thread_specific<HasSpecialAndMoveCtor, Allocator<HasSpecialAndMoveCtor> > ets5(SecretTag);
    ets5.local();
    ets5.combine_each(EmptyCombineEach<HasSpecialAndMoveCtor>());
    // Test that move operations do not impose extra requirements
    // Default allocator is used. If it does not match Allocator, there will be elementwise move
    tbb::enumerable_thread_specific<HasSpecialAndMoveCtor> ets6( std::move(ets4) );
    ets6.combine_each(EmptyCombineEach<HasSpecialAndMoveCtor>());
    ets6 = std::move(ets5);
#endif
}

class BigType {
public:
    BigType() { /* avoid cl warning C4345 about default initialization of POD types */ }
    char my_data[12 * 1024 * 1024];
};

template<template<class> class Allocator>
void TestConstructorWithBigType(const char *allocator_name) {
    typedef tbb::enumerable_thread_specific<BigType, Allocator<BigType> > CounterBigType;
    REMARK("TestConstructorWithBigType<%s>\n", allocator_name);
    // Test default constructor
    CounterBigType MyCounters;
    // Create a local instance.
    typename CounterBigType::reference my_local = MyCounters.local();
    my_local.my_data[0] = 'a';
    // Test copy constructor
    CounterBigType MyCounters2(MyCounters);
    ASSERT(check_alignment(MyCounters2.local(), allocator_name).my_data[0]=='a', NULL);
}

int TestMain () {
    size_t tbb_allocator_mask;
    size_t cache_allocator_mask = tbb::internal::NFS_GetLineSize();
    REMARK("estimatedCacheLineSize == %d, NFS_GetLineSize() returns %d\n",
                (int)estimatedCacheLineSize, (int)tbb::internal::NFS_GetLineSize());
    //TODO: use __TBB_alignof(T) to check for local() results instead of using internal knowledges of ets element padding
    if(tbb::tbb_allocator<int>::allocator_type() == tbb::tbb_allocator<int>::standard) {
        // scalable allocator is not available.
        tbb_allocator_mask = 1;
        REMARK("tbb::tbb_allocator is not available\n");
    }
    else {
        // this value is for large objects, but will be correct for small.
        tbb_allocator_mask = estimatedCacheLineSize;
    }
    AlignMask = cache_allocator_mask;
    TestInstantiation<tbb::cache_aligned_allocator>("tbb::cache_aligned_allocator");
    AlignMask = tbb_allocator_mask;
    TestInstantiation<tbb::tbb_allocator>("tbb::tbb_allocator");
    AlignMask = cache_allocator_mask;
    run_assignment_and_copy_constructor_tests<tbb::cache_aligned_allocator>("tbb::cache_aligned_allocator");
    AlignMask = tbb_allocator_mask;
    run_assignment_and_copy_constructor_tests<tbb::tbb_allocator>("tbb::tbb_allocator");
    run_segmented_iterator_tests();
    flog_key_creation_and_deletion();

    if (MinThread == 0) {
        run_serial_tests();
        MinThread = 1;
    }
    if (MaxThread > 0) {
        AlignMask = cache_allocator_mask;
        run_parallel_tests<tbb::cache_aligned_allocator>("tbb::cache_aligned_allocator");
        AlignMask = tbb_allocator_mask;
        run_parallel_tests<tbb::tbb_allocator>("tbb::tbb_allocator");
        run_cross_type_tests();
    }

    AlignMask = cache_allocator_mask;
    TestConstructorWithBigType<tbb::cache_aligned_allocator>("tbb::cache_aligned_allocator");
    AlignMask = tbb_allocator_mask;
    TestConstructorWithBigType<tbb::tbb_allocator>("tbb::tbb_allocator");

    allocate_ets_element_on_stack<int,char>("int vs. char");
    allocate_ets_element_on_stack<int,short>("int vs. short");
    allocate_ets_element_on_stack<int,char[3]>("int vs. char[3]");
    allocate_ets_element_on_stack<float,char>("float vs. char");
    allocate_ets_element_on_stack<float,short>("float vs. short");
    allocate_ets_element_on_stack<float,char[3]>("float vs. char[3]");

    return Harness::Done;
}
