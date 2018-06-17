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

#if _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "tbb/concurrent_vector.h"
#include "tbb/tbb_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_exception.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <numeric>
#include "harness_report.h"
#include "harness_assert.h"
#include "harness_allocator.h"
#include "harness_defs.h"
#include "test_container_move_support.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
    #pragma warning (disable: 4800)
#endif

#if TBB_USE_EXCEPTIONS
static bool known_issue_verbose = false;
#define KNOWN_ISSUE(msg) if(!known_issue_verbose) known_issue_verbose = true, REPORT(msg)
#endif /* TBB_USE_EXCEPTIONS */

inline void NextSize( int& s ) {
    if( s<=32 ) ++s;
    else s += s/10;
}

//! Check vector have expected size and filling
template<typename vector_t>
static void CheckVector( const vector_t& cv, size_t expected_size, size_t old_size ) {
    ASSERT( cv.capacity()>=expected_size, NULL );
    ASSERT( cv.size()==expected_size, NULL );
    ASSERT( cv.empty()==(expected_size==0), NULL );
    for( int j=0; j<int(expected_size); ++j ) {
        if( cv[j].bar()!=~j )
            REPORT("ERROR on line %d for old_size=%ld expected_size=%ld j=%d\n",__LINE__,long(old_size),long(expected_size),j);
    }
}

//! Test of assign, grow, copying with various sizes
void TestResizeAndCopy() {
    typedef static_counting_allocator<debug_allocator<Foo,std::allocator>, std::size_t> allocator_t;
    typedef tbb::concurrent_vector<Foo, allocator_t> vector_t;
    allocator_t::init_counters();
    for( int old_size=0; old_size<=128; NextSize( old_size ) ) {
        for( int new_size=0; new_size<=1280; NextSize( new_size ) ) {
            size_t count = FooCount;
            vector_t v;
            ASSERT( count==FooCount, NULL );
            v.assign(old_size/2, Foo() );
            ASSERT( count+old_size/2==FooCount, NULL );
            for( int j=0; j<old_size/2; ++j )
                ASSERT( v[j].state == Foo::CopyInitialized, NULL);
            v.assign(FooIterator(0), FooIterator(old_size));
            v.resize(new_size, Foo(33) );
            ASSERT( count+new_size==FooCount, NULL );
            for( int j=0; j<new_size; ++j ) {
                int expected = j<old_size ? j : 33;
                if( v[j].bar()!=expected )
                    REPORT("ERROR on line %d for old_size=%ld new_size=%ld v[%ld].bar()=%d != %d\n",__LINE__,long(old_size),long(new_size),long(j),v[j].bar(), expected);
            }
            ASSERT( v.size()==size_t(new_size), NULL );
            for( int j=0; j<new_size; ++j ) {
                v[j].bar() = ~j;
            }
            const vector_t& cv = v;
            // Try copy constructor
            vector_t copy_of_v(cv);
            CheckVector(cv,new_size,old_size);
            ASSERT( !(v != copy_of_v), NULL );
            v.clear();
            ASSERT( v.empty(), NULL );
            swap(v, copy_of_v);
            ASSERT( copy_of_v.empty(), NULL );
            CheckVector(v,new_size,old_size);
        }
    }
    ASSERT( allocator_t::items_allocated == allocator_t::items_freed, NULL);
    ASSERT( allocator_t::allocations == allocator_t::frees, NULL);
}

//! Test reserve, compact, capacity
void TestCapacity() {
    typedef static_counting_allocator<debug_allocator<Foo,tbb::cache_aligned_allocator>, std::size_t> allocator_t;
    typedef tbb::concurrent_vector<Foo, allocator_t> vector_t;
    allocator_t::init_counters();
    for( size_t old_size=0; old_size<=11000; old_size=(old_size<5 ? old_size+1 : 3*old_size) ) {
        for( size_t new_size=0; new_size<=11000; new_size=(new_size<5 ? new_size+1 : 3*new_size) ) {
            size_t count = FooCount;
            {
                vector_t v; v.reserve(old_size);
                ASSERT( v.capacity()>=old_size, NULL );
                v.reserve( new_size );
                ASSERT( v.capacity()>=old_size, NULL );
                ASSERT( v.capacity()>=new_size, NULL );
                ASSERT( v.empty(), NULL );
                size_t fill_size = 2*new_size;
                for( size_t i=0; i<fill_size; ++i ) {
                    ASSERT( size_t(FooCount)==count+i, NULL );
                    size_t j = v.grow_by(1) - v.begin();
                    ASSERT( j==i, NULL );
                    v[j].bar() = int(~j);
                }
                vector_t copy_of_v(v); // should allocate first segment with same size as for shrink_to_fit()
                if(__TBB_Log2(/*reserved size*/old_size|1) > __TBB_Log2(fill_size|1) )
                    ASSERT( v.capacity() != copy_of_v.capacity(), NULL );
                v.shrink_to_fit();
                ASSERT( v.capacity() == copy_of_v.capacity(), NULL );
                CheckVector(v, new_size*2, old_size); // check vector correctness
                ASSERT( v==copy_of_v, NULL ); // TODO: check also segments layout equality
            }
            ASSERT( FooCount==count, NULL );
        }
    }
    ASSERT( allocator_t::items_allocated == allocator_t::items_freed, NULL);
    ASSERT( allocator_t::allocations == allocator_t::frees, NULL);
}

struct AssignElement {
    typedef tbb::concurrent_vector<int>::range_type::iterator iterator;
    iterator base;
    void operator()( const tbb::concurrent_vector<int>::range_type& range ) const {
        for( iterator i=range.begin(); i!=range.end(); ++i ) {
            if( *i!=0 )
                REPORT("ERROR for v[%ld]\n", long(i-base));
            *i = int(i-base);
        }
    }
    AssignElement( iterator base_ ) : base(base_) {}
};

struct CheckElement {
    typedef tbb::concurrent_vector<int>::const_range_type::iterator iterator;
    iterator base;
    void operator()( const tbb::concurrent_vector<int>::const_range_type& range ) const {
        for( iterator i=range.begin(); i!=range.end(); ++i )
            if( *i != int(i-base) )
                REPORT("ERROR for v[%ld]\n", long(i-base));
    }
    CheckElement( iterator base_ ) : base(base_) {}
};

#include "tbb/tick_count.h"
#include "tbb/parallel_for.h"
#include "harness.h"

//! Problem size
const size_t N = 500000;

//! Test parallel access by iterators
void TestParallelFor( int nthread ) {
    typedef tbb::concurrent_vector<int> vector_t;
    vector_t v;
    v.resize(N);
    tbb::tick_count t0 = tbb::tick_count::now();
    REMARK("Calling parallel_for with %ld threads\n",long(nthread));
    tbb::parallel_for( v.range(10000), AssignElement(v.begin()) );
    tbb::tick_count t1 = tbb::tick_count::now();
    const vector_t& u = v;
    tbb::parallel_for( u.range(10000), CheckElement(u.begin()) );
    tbb::tick_count t2 = tbb::tick_count::now();
    REMARK("Time for parallel_for: assign time = %8.5f, check time = %8.5f\n",
               (t1-t0).seconds(),(t2-t1).seconds());
    for( long i=0; size_t(i)<v.size(); ++i )
        if( v[i]!=i )
            REPORT("ERROR for v[%ld]\n", i);
}

template<typename Iterator1, typename Iterator2>
void TestIteratorAssignment( Iterator2 j ) {
    Iterator1 i(j);
    ASSERT( i==j, NULL );
    ASSERT( !(i!=j), NULL );
    Iterator1 k;
    k = j;
    ASSERT( k==j, NULL );
    ASSERT( !(k!=j), NULL );
}

template<typename Range1, typename Range2>
void TestRangeAssignment( Range2 r2 ) {
    Range1 r1(r2); r1 = r2;
}

template<typename Iterator, typename T>
void TestIteratorTraits() {
    AssertSameType( static_cast<typename Iterator::difference_type*>(0), static_cast<ptrdiff_t*>(0) );
    AssertSameType( static_cast<typename Iterator::value_type*>(0), static_cast<T*>(0) );
    AssertSameType( static_cast<typename Iterator::pointer*>(0), static_cast<T**>(0) );
    AssertSameType( static_cast<typename Iterator::iterator_category*>(0), static_cast<std::random_access_iterator_tag*>(0) );
    T x;
    typename Iterator::reference xr = x;
    typename Iterator::pointer xp = &x;
    ASSERT( &xr==xp, NULL );
}

template<typename Vector, typename Iterator>
void CheckConstIterator( const Vector& u, int i, const Iterator& cp ) {
    typename Vector::const_reference pref = *cp;
    if( pref.bar()!=i )
        REPORT("ERROR for u[%ld] using const_iterator\n", long(i));
    typename Vector::difference_type delta = cp-u.begin();
    ASSERT( delta==i, NULL );
    if( u[i].bar()!=i )
        REPORT("ERROR for u[%ld] using subscripting\n", long(i));
    ASSERT( u.begin()[i].bar()==i, NULL );
}

template<typename Iterator1, typename Iterator2, typename V>
void CheckIteratorComparison( V& u ) {
    V u2 = u;
    Iterator1 i = u.begin();

    for( int i_count=0; i_count<100; ++i_count ) {
        Iterator2 j = u.begin();
        Iterator2 i2 = u2.begin();
        for( int j_count=0; j_count<100; ++j_count ) {
            ASSERT( (i==j)==(i_count==j_count), NULL );
            ASSERT( (i!=j)==(i_count!=j_count), NULL );
            ASSERT( (i-j)==(i_count-j_count), NULL );
            ASSERT( (i<j)==(i_count<j_count), NULL );
            ASSERT( (i>j)==(i_count>j_count), NULL );
            ASSERT( (i<=j)==(i_count<=j_count), NULL );
            ASSERT( (i>=j)==(i_count>=j_count), NULL );
            ASSERT( !(i==i2), NULL );
            ASSERT( i!=i2, NULL );
            ++j;
            ++i2;
        }
        ++i;
    }
}

template<typename Vector, typename T>
void TestGrowToAtLeastWithSourceParameter(T const& src){
    static const size_t vector_size = 10;
    Vector v1(vector_size,src);
    Vector v2;
    v2.grow_to_at_least(vector_size,src);
    ASSERT(v1==v2,"grow_to_at_least(vector_size,src) did not properly initialize new elements ?");
}
//! Test sequential iterators for vector type V.
/** Also does timing. */
template<typename T>
void TestSequentialFor() {
    typedef tbb::concurrent_vector<FooWithAssign> V;
    V v(N);
    ASSERT(v.grow_by(0) == v.grow_by(0, FooWithAssign()), NULL);

    // Check iterator
    tbb::tick_count t0 = tbb::tick_count::now();
    typename V::iterator p = v.begin();
    ASSERT( !(*p).is_const(), NULL );
    ASSERT( !p->is_const(), NULL );
    for( int i=0; size_t(i)<v.size(); ++i, ++p ) {
        if( (*p).state!=Foo::DefaultInitialized )
            REPORT("ERROR for v[%ld]\n", long(i));
        typename V::reference pref = *p;
        pref.bar() = i;
        typename V::difference_type delta = p-v.begin();
        ASSERT( delta==i, NULL );
        ASSERT( -delta<=0, "difference type not signed?" );
    }
    tbb::tick_count t1 = tbb::tick_count::now();

    // Check const_iterator going forwards
    const V& u = v;
    typename V::const_iterator cp = u.begin();
    ASSERT( cp == v.cbegin(), NULL );
    ASSERT( (*cp).is_const(), NULL );
    ASSERT( cp->is_const(), NULL );
    ASSERT( *cp == v.front(), NULL);
    for( int i=0; size_t(i)<u.size(); ++i ) {
        CheckConstIterator(u,i,cp);
        V::const_iterator &cpr = ++cp;
        ASSERT( &cpr == &cp, "pre-increment not returning a reference?");
    }
    tbb::tick_count t2 = tbb::tick_count::now();
    REMARK("Time for serial for:  assign time = %8.5f, check time = %8.5f\n",
               (t1-t0).seconds(),(t2-t1).seconds());

    // Now go backwards
    cp = u.end();
    ASSERT( cp == v.cend(), NULL );
    for( int i=int(u.size()); i>0; ) {
        --i;
        V::const_iterator &cpr = --cp;
        ASSERT( &cpr == &cp, "pre-decrement not returning a reference?");
        if( i>0 ) {
            typename V::const_iterator cp_old = cp--;
            intptr_t here = (*cp_old).bar();
            ASSERT( here==u[i].bar(), NULL );
            typename V::const_iterator cp_new = cp++;
            intptr_t prev = (*cp_new).bar();
            ASSERT( prev==u[i-1].bar(), NULL );
        }
        CheckConstIterator(u,i,cp);
    }

    // Now go forwards and backwards
    ptrdiff_t k = 0;
    cp = u.begin();
    for( size_t i=0; i<u.size(); ++i ) {
        CheckConstIterator(u,int(k),cp);
        typename V::difference_type delta = i*3 % u.size();
        if( 0<=k+delta && size_t(k+delta)<u.size() ) {
            V::const_iterator &cpr = (cp += delta);
            ASSERT( &cpr == &cp, "+= not returning a reference?");
            k += delta;
        }
        delta = i*7 % u.size();
        if( 0<=k-delta && size_t(k-delta)<u.size() ) {
            if( i&1 ) {
                V::const_iterator &cpr = (cp -= delta);
                ASSERT( &cpr == &cp, "-= not returning a reference?");
            } else
                cp = cp - delta;        // Test operator-
            k -= delta;
        }
    }

    for( int i=0; size_t(i)<u.size(); i=(i<50?i+1:i*3) )
        for( int j=-i; size_t(i+j)<u.size(); j=(j<50?j+1:j*5) ) {
            ASSERT( (u.begin()+i)[j].bar()==i+j, NULL );
            ASSERT( (v.begin()+i)[j].bar()==i+j, NULL );
            ASSERT((v.cbegin()+i)[j].bar()==i+j, NULL );
            ASSERT( (i+u.begin())[j].bar()==i+j, NULL );
            ASSERT( (i+v.begin())[j].bar()==i+j, NULL );
            ASSERT((i+v.cbegin())[j].bar()==i+j, NULL );
        }

    CheckIteratorComparison<typename V::iterator, typename V::iterator>(v);
    CheckIteratorComparison<typename V::iterator, typename V::const_iterator>(v);
    CheckIteratorComparison<typename V::const_iterator, typename V::iterator>(v);
    CheckIteratorComparison<typename V::const_iterator, typename V::const_iterator>(v);

    TestIteratorAssignment<typename V::const_iterator>( u.begin() );
    TestIteratorAssignment<typename V::const_iterator>( v.begin() );
    TestIteratorAssignment<typename V::const_iterator>( v.cbegin() );
    TestIteratorAssignment<typename V::iterator>( v.begin() );
    // doesn't compile as expected: TestIteratorAssignment<typename V::iterator>( u.begin() );

    TestRangeAssignment<typename V::const_range_type>( u.range() );
    TestRangeAssignment<typename V::const_range_type>( v.range() );
    TestRangeAssignment<typename V::range_type>( v.range() );
    // doesn't compile as expected: TestRangeAssignment<typename V::range_type>( u.range() );

    // Check reverse_iterator
    typename V::reverse_iterator rp = v.rbegin();
    for( size_t i=v.size(); i>0; --i, ++rp ) {
        typename V::reference pref = *rp;
        ASSERT( size_t(pref.bar())==i-1, NULL );
        ASSERT( rp!=v.rend(), NULL );
    }
    ASSERT( rp==v.rend(), NULL );

    // Check const_reverse_iterator
    typename V::const_reverse_iterator crp = u.rbegin();
    ASSERT( crp == v.crbegin(), NULL );
    ASSERT( *crp == v.back(), NULL);
    for( size_t i=v.size(); i>0; --i, ++crp ) {
        typename V::const_reference cpref = *crp;
        ASSERT( size_t(cpref.bar())==i-1, NULL );
        ASSERT( crp!=u.rend(), NULL );
    }
    ASSERT( crp == u.rend(), NULL );
    ASSERT( crp == v.crend(), NULL );

    TestIteratorAssignment<typename V::const_reverse_iterator>( u.rbegin() );
    TestIteratorAssignment<typename V::reverse_iterator>( v.rbegin() );

    // test compliance with C++ Standard 2003, clause 23.1.1p9
    {
        tbb::concurrent_vector<int> v1, v2(1, 100);
        v1.assign(1, 100); ASSERT(v1 == v2, NULL);
        ASSERT(v1.size() == 1 && v1[0] == 100, "used integral iterators");
    }

    // cross-allocator tests
#if !defined(_WIN64) || defined(_CPPLIB_VER)
    typedef local_counting_allocator<std::allocator<int>, size_t> allocator1_t;
    typedef tbb::cache_aligned_allocator<void> allocator2_t;
    typedef tbb::concurrent_vector<FooWithAssign, allocator1_t> V1;
    typedef tbb::concurrent_vector<FooWithAssign, allocator2_t> V2;
    V1 v1( v ); // checking cross-allocator copying
    V2 v2( 10 ); v2 = v1; // checking cross-allocator assignment
    ASSERT( (v1 == v) && !(v2 != v), NULL);
    ASSERT( !(v1 < v) && !(v2 > v), NULL);
    ASSERT( (v1 <= v) && (v2 >= v), NULL);
#endif
}

namespace test_grow_to_at_least_helpers {
    template<typename MyVector >
    class GrowToAtLeast: NoAssign {
        typedef typename MyVector::const_reference const_reference;

        const bool my_use_two_args_form ;
        MyVector& my_vector;
        const_reference my_init_from;
    public:
        void operator()( const tbb::blocked_range<size_t>& range ) const {
            for( size_t i=range.begin(); i!=range.end(); ++i ) {
                size_t n = my_vector.size();
                size_t req = (i % (2*n+1))+1;

                typename MyVector::iterator p;
                Foo::State desired_state;
                if (my_use_two_args_form){
                    p = my_vector.grow_to_at_least(req,my_init_from);
                    desired_state = Foo::CopyInitialized;
                }else{
                    p = my_vector.grow_to_at_least(req);
                    desired_state = Foo::DefaultInitialized;
                }
                if( p-my_vector.begin() < typename MyVector::difference_type(req) )
                    ASSERT( p->state == desired_state || p->state == Foo::ZeroInitialized, NULL );
                ASSERT( my_vector.size()>=req, NULL );
            }
        }
        GrowToAtLeast(bool use_two_args_form, MyVector& vector, const_reference init_from )
            : my_use_two_args_form(use_two_args_form), my_vector(vector), my_init_from(init_from) {}
    };
}

template<bool use_two_arg_form>
void TestConcurrentGrowToAtLeastImpl() {
    using namespace test_grow_to_at_least_helpers;
    typedef static_counting_allocator< tbb::zero_allocator<Foo> > MyAllocator;
    typedef tbb::concurrent_vector<Foo, MyAllocator> MyVector;
    Foo copy_from;
    MyAllocator::init_counters();
    MyVector v(2, Foo(), MyAllocator());
    for( size_t s=1; s<1000; s*=10 ) {
        tbb::parallel_for( tbb::blocked_range<size_t>(0,10000*s,s), GrowToAtLeast<MyVector>(use_two_arg_form, v, copy_from), tbb::simple_partitioner() );
    }
    v.clear();
    ASSERT( 0 == v.get_allocator().frees, NULL);
    v.shrink_to_fit();
    size_t items_allocated = v.get_allocator().items_allocated,
           items_freed = v.get_allocator().items_freed;
    size_t allocations = v.get_allocator().allocations,
           frees = v.get_allocator().frees;
    ASSERT( items_allocated == items_freed, NULL);
    ASSERT( allocations == frees, NULL);
}

void TestConcurrentGrowToAtLeast() {
    TestConcurrentGrowToAtLeastImpl<false>();
    TestConcurrentGrowToAtLeastImpl<true>();
}

struct grain_map: NoAssign {
    enum grow_method_enum {
        grow_by_range = 1,
        grow_by_default,
        grow_by_copy,
        grow_by_init_list,
        push_back,
        push_back_move,
        emplace_back,
        last_method
    };

    struct range_part {
        size_t number_of_parts;
        grain_map::grow_method_enum method;
        bool distribute;
        Foo::State expected_element_state;
    };

    const std::vector<range_part> distributed;
    const std::vector<range_part> batched;
    const size_t total_number_of_parts;

    grain_map(const range_part* begin, const range_part* end)
    : distributed(separate(begin,end, &distributed::is_not))
    , batched(separate(begin,end, &distributed::is_yes))
    , total_number_of_parts(std::accumulate(begin, end, (size_t)0, &sum_number_of_parts::sum))
    {}

private:
    struct sum_number_of_parts{
        static size_t sum(size_t accumulator, grain_map::range_part const& rp){ return accumulator + rp.number_of_parts;}
    };

    template <typename functor_t>
    static std::vector<range_part> separate(const range_part* begin, const range_part* end, functor_t f){
        std::vector<range_part> part;
        part.reserve(std::distance(begin,end));
        //copy all that false==f(*it)
        std::remove_copy_if(begin, end, std::back_inserter(part), f);

        return part;
    }

    struct distributed {
        static bool is_not(range_part const& rp){ return !rp.distribute;}
        static bool is_yes(range_part const& rp){ return rp.distribute;}
    };
};

//! Test concurrent invocations of method concurrent_vector::grow_by
template<typename MyVector>
class GrowBy: NoAssign {
    MyVector& my_vector;
    const grain_map& my_grain_map;
    size_t my_part_weight;
public:
    void operator()( const tbb::blocked_range<size_t>& range ) const {
        ASSERT( range.begin() < range.end(), NULL );

        size_t current_adding_index_in_cvector = range.begin();

        for(size_t index=0; index < my_grain_map.batched.size(); ++index){
            const grain_map::range_part& batch_part = my_grain_map.batched[index];
            const size_t number_of_items_to_add = batch_part.number_of_parts * my_part_weight;
            const size_t end = current_adding_index_in_cvector + number_of_items_to_add;

            switch(batch_part.method){
            case grain_map::grow_by_range : {
                    my_vector.grow_by(FooIterator(current_adding_index_in_cvector),FooIterator(end));
                } break;
            case grain_map::grow_by_default : {
                    typename MyVector::iterator const s = my_vector.grow_by(number_of_items_to_add);
                    for( size_t k = 0; k < number_of_items_to_add; ++k )
                        s[k].bar() = current_adding_index_in_cvector + k;
                } break;
#if __TBB_INITIALIZER_LISTS_PRESENT
            case grain_map::grow_by_init_list : {
                    FooIterator curr(current_adding_index_in_cvector);
                    for ( size_t k = 0; k < number_of_items_to_add; ++k ) {
                        if ( k + 4 < number_of_items_to_add ) {
                            my_vector.grow_by( { *curr++, *curr++, *curr++, *curr++, *curr++ } );
                            k += 4;
                        } else {
                            my_vector.grow_by( { *curr++ } );
                        }
                    }
                    ASSERT( curr == FooIterator(end), NULL );
                } break;
#endif
            default : { ASSERT(false, "using unimplemented method of batch add in ConcurrentGrow test.");} break;
            };

            current_adding_index_in_cvector = end;
        }

        std::vector<size_t> items_left_to_add(my_grain_map.distributed.size());
        for (size_t i=0; i<my_grain_map.distributed.size(); ++i ){
            items_left_to_add[i] = my_grain_map.distributed[i].number_of_parts * my_part_weight;
        }

        for (;current_adding_index_in_cvector < range.end(); ++current_adding_index_in_cvector){
            size_t method_index = current_adding_index_in_cvector % my_grain_map.distributed.size();

            if (! items_left_to_add[method_index]) {
                struct not_zero{
                    static bool is(size_t items_to_add){ return items_to_add;}
                };
                method_index = std::distance(items_left_to_add.begin(), std::find_if(items_left_to_add.begin(), items_left_to_add.end(), &not_zero::is));
                ASSERT(method_index < my_grain_map.distributed.size(), "incorrect test setup - wrong expected distribution: left free space but no elements to add?");
            };

            ASSERT(items_left_to_add[method_index], "logic error ?");
            const grain_map::range_part& distributed_part = my_grain_map.distributed[method_index];

            typename MyVector::iterator r;
            typename MyVector::value_type source;
            source.bar() = current_adding_index_in_cvector;

            switch(distributed_part.method){
            case grain_map::grow_by_default : {
                    (r = my_vector.grow_by(1))->bar() = current_adding_index_in_cvector;
                } break;
            case grain_map::grow_by_copy : {
                    r = my_vector.grow_by(1, source);
                } break;
            case grain_map::push_back : {
                    r = my_vector.push_back(source);
                } break;
#if __TBB_CPP11_RVALUE_REF_PRESENT
            case grain_map::push_back_move : {
                    r = my_vector.push_back(std::move(source));
                } break;
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
            case grain_map::emplace_back : {
                    r = my_vector.emplace_back(current_adding_index_in_cvector);
                } break;
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

            default : { ASSERT(false, "using unimplemented method of batch add in ConcurrentGrow test.");} break;
            };

            ASSERT( static_cast<size_t>(r->bar()) == current_adding_index_in_cvector, NULL );
            }
        }

    GrowBy( MyVector& vector, const grain_map& m, size_t part_weight )
    : my_vector(vector)
    , my_grain_map(m)
    , my_part_weight(part_weight)
    {
    }
};

const grain_map::range_part concurrent_grow_single_range_map [] = {
//  number_of_parts,         method,             distribute,   expected_element_state
        {3,           grain_map::grow_by_range,     false,
                                                            #if  __TBB_CPP11_RVALUE_REF_PRESENT
                                                                Foo::MoveInitialized
                                                            #else
                                                                Foo::CopyInitialized
                                                            #endif
        },
#if __TBB_INITIALIZER_LISTS_PRESENT && !__TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
        {1,           grain_map::grow_by_init_list, false,   Foo::CopyInitialized},
#endif
        {2,           grain_map::grow_by_default,   false,   Foo::DefaultInitialized},
        {1,           grain_map::grow_by_default,   true,    Foo::DefaultInitialized},
        {1,           grain_map::grow_by_copy,      true,    Foo::CopyInitialized},
        {1,           grain_map::push_back,         true,    Foo::CopyInitialized},
#if __TBB_CPP11_RVALUE_REF_PRESENT
        {1,           grain_map::push_back_move,    true,    Foo::MoveInitialized},
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
        {1,           grain_map::emplace_back,      true,    Foo::DirectInitialized},
#endif // __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT
};

//! Test concurrent invocations of grow methods
void TestConcurrentGrowBy( int nthread ) {

    typedef static_counting_allocator<debug_allocator<Foo> > MyAllocator;
    typedef tbb::concurrent_vector<Foo, MyAllocator> MyVector;

#if __TBB_INITIALIZER_LISTS_PRESENT && __TBB_CPP11_INIT_LIST_TEMP_OBJS_LIFETIME_BROKEN
    static bool is_reported = false;
    if ( !is_reported ) {
        REPORT( "Known issue: concurrent tests of grow_by(std::initializer_list) are skipped.\n" );
        is_reported = true;
    }
#endif

    MyAllocator::init_counters();
    {
        grain_map m(concurrent_grow_single_range_map, Harness::end(concurrent_grow_single_range_map));

        static const size_t desired_grain_size = 100;

        static const size_t part_weight = desired_grain_size / m.total_number_of_parts;
        static const size_t grain_size = part_weight * m.total_number_of_parts;
        static const size_t number_of_grains = 8; //this should be (power of two) in order to get minimal ranges equal to grain_size
        static const size_t range_size = grain_size * number_of_grains;

        MyAllocator a;
        MyVector v( a );
        tbb::parallel_for( tbb::blocked_range<size_t>(0,range_size,grain_size), GrowBy<MyVector>(v, m, part_weight), tbb::simple_partitioner() );
        ASSERT( v.size()==size_t(range_size), NULL );

        // Verify that v is a permutation of 0..m
        size_t inversions = 0, direct_inits = 0, def_inits = 0, copy_inits = 0, move_inits = 0;
        std::vector<bool> found(range_size, 0);
        for( size_t i=0; i<range_size; ++i ) {
            if( v[i].state == Foo::DefaultInitialized ) ++def_inits;
            else if( v[i].state == Foo::DirectInitialized ) ++direct_inits;
            else if( v[i].state == Foo::CopyInitialized ) ++copy_inits;
            else if( v[i].state == Foo::MoveInitialized ) ++move_inits;
            else {
                REMARK("i: %d ", i);
                ASSERT( false, "v[i] seems not initialized");
            }
            intptr_t index = v[i].bar();
            ASSERT( !found[index], NULL );
            found[index] = true;
            if( i>0 )
                inversions += v[i].bar()<v[i-1].bar();
        }
        for( size_t i=0; i<range_size; ++i ) {
            ASSERT( found[i], NULL );
            ASSERT( nthread>1 || v[i].bar() == static_cast<intptr_t>(i), "sequential execution is wrong" );
        }

        REMARK("Initialization by default constructor: %d, by copy: %d, by move: %d\n", def_inits, copy_inits, move_inits);

        size_t expected_direct_inits = 0, expected_def_inits = 0, expected_copy_inits = 0, expected_move_inits = 0;
        for (size_t i=0; i<Harness::array_length(concurrent_grow_single_range_map); ++i){
            const grain_map::range_part& rp =concurrent_grow_single_range_map[i];
            switch (rp.expected_element_state){
            case Foo::DefaultInitialized: { expected_def_inits += rp.number_of_parts ; } break;
            case Foo::DirectInitialized:  { expected_direct_inits += rp.number_of_parts ;} break;
            case Foo::MoveInitialized:    { expected_move_inits += rp.number_of_parts ;} break;
            case Foo::CopyInitialized:    { expected_copy_inits += rp.number_of_parts ;} break;
            default: {ASSERT(false, "unexpected expected state");}break;
            };
        }

        expected_def_inits    *= part_weight * number_of_grains;
        expected_move_inits   *= part_weight * number_of_grains;
        expected_copy_inits   *= part_weight * number_of_grains;
        expected_direct_inits *= part_weight * number_of_grains;

        ASSERT( def_inits == expected_def_inits , NULL);
        ASSERT( copy_inits == expected_copy_inits , NULL);
        ASSERT( move_inits == expected_move_inits , NULL);
        ASSERT( direct_inits == expected_direct_inits , NULL);

        if( nthread>1 && inversions<range_size/20 )
            REPORT("Warning: not much concurrency in TestConcurrentGrowBy (%d inversions)\n", inversions);
    }
    //TODO: factor this into separate thing, as it seems to used in big number of tests
    size_t items_allocated = MyAllocator::items_allocated,
           items_freed = MyAllocator::items_freed;
    size_t allocations = MyAllocator::allocations,
           frees = MyAllocator::frees;
    ASSERT( items_allocated == items_freed, NULL);
    ASSERT( allocations == frees, NULL);
}

template <typename Vector>
void test_grow_by_empty_range( Vector &v, typename Vector::value_type* range_begin_end ) {
    const Vector v_copy = v;
    ASSERT( v.grow_by( range_begin_end, range_begin_end ) == v.end(), "grow_by(empty_range) returned a wrong iterator." );
    ASSERT( v == v_copy, "grow_by(empty_range) has changed the vector." );
}

void TestSerialGrowByRange( bool fragmented_vector ) {
    tbb::concurrent_vector<int> v;
    if ( fragmented_vector ) {
        v.reserve( 1 );
    }
    int init_range[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    ASSERT( v.grow_by( init_range, init_range + (Harness::array_length( init_range )) ) == v.begin(), "grow_by(I,I) returned a wrong iterator." );
    ASSERT( std::equal( v.begin(), v.end(), init_range ), "grow_by(I,I) did not properly copied all elements ?" );
    test_grow_by_empty_range( v, init_range );
    test_grow_by_empty_range( v, (int*)NULL );
}

//TODO: move this to more appropriate place, smth like test_harness.cpp
void TestArrayLength(){
    int five_element_array[5] = {0};
    ASSERT(Harness::array_length(five_element_array)==5,"array_length failed to determine length of non empty non dynamic array");
}

#if __TBB_INITIALIZER_LISTS_PRESENT
#include "test_initializer_list.h"

struct test_grow_by {
    template<typename container_type, typename element_type>
    static void do_test( std::initializer_list<element_type> const& il, container_type const& expected ) {
        container_type vd;
        vd.grow_by( il );
        ASSERT( vd == expected, "grow_by with an initializer list failed" );
    }
};

void TestInitList() {
    REMARK( "testing initializer_list methods \n" );
    using namespace initializer_list_support_tests;
    TestInitListSupport<tbb::concurrent_vector<char>, test_grow_by>( { 1, 2, 3, 4, 5 } );
    TestInitListSupport<tbb::concurrent_vector<int>, test_grow_by>( {} );
}
#endif //if __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_RANGE_BASED_FOR_PRESENT
#include "test_range_based_for.h"

void TestRangeBasedFor(){
    using namespace range_based_for_support_tests;

    REMARK("testing range based for loop compatibility \n");
    typedef tbb::concurrent_vector<int> c_vector;
    c_vector a_c_vector;

    const int sequence_length = 100;
    for (int i =1; i<= sequence_length; ++i){
        a_c_vector.push_back(i);
    }

    ASSERT( range_based_for_accumulate(a_c_vector, std::plus<int>(), 0) == gauss_summ_of_int_sequence(sequence_length), "incorrect accumulated value generated via range based for ?");
}
#endif //if __TBB_RANGE_BASED_FOR_PRESENT

#if TBB_USE_EXCEPTIONS
#endif //TBB_USE_EXCEPTIONS

#if __TBB_CPP11_RVALUE_REF_PRESENT
namespace move_semantics_helpers{
    struct move_only_type:NoCopy{
        const int* my_pointer;
        move_only_type(move_only_type && other): my_pointer(other.my_pointer){other.my_pointer=NULL;}
        explicit move_only_type(const int* value): my_pointer(value) {}
    };
}

void TestPushBackMoveOnlyContainee(){
    using namespace move_semantics_helpers;
    typedef tbb::concurrent_vector<move_only_type > vector_t;
    vector_t v;
    static const int magic_number =7;
    move_only_type src(&magic_number);
    v.push_back(std::move(src));
    ASSERT(v[0].my_pointer == &magic_number,"item was incorrectly moved during push_back?");
    ASSERT(src.my_pointer == NULL,"item was incorrectly moved during push_back?");
}

namespace emplace_helpers{
    struct wrapper_type:NoCopy{
        int value1;
        int value2;
        explicit wrapper_type(int v1, int v2) : value1 (v1), value2(v2) {}
        friend bool operator==(const wrapper_type& lhs, const wrapper_type& rhs){
            return (lhs.value1 == rhs.value1) && (lhs.value2 == rhs.value2 );
        }
    };
}
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
//TODO: extend the test to number of types e.g. std::string
void TestEmplaceBack(){
    using namespace emplace_helpers;
    typedef tbb::concurrent_vector<wrapper_type > vector_t;
    vector_t v;
    v.emplace_back(1,2);
    ASSERT(v[0] == wrapper_type(1,2),"incorrectly in-place constructed item during emplace_back?");
}
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

//! Test the assignment operator and swap
void TestAssign() {
    typedef tbb::concurrent_vector<FooWithAssign, local_counting_allocator<std::allocator<FooWithAssign>, size_t > > vector_t;
    local_counting_allocator<std::allocator<FooWithAssign>, size_t > init_alloc;
    init_alloc.allocations = 100;
    for( int dst_size=1; dst_size<=128; NextSize( dst_size ) ) {
        for( int src_size=2; src_size<=128; NextSize( src_size ) ) {
            vector_t u(FooIterator(0), FooIterator(src_size), init_alloc);
            for( int i=0; i<src_size; ++i )
                ASSERT( u[i].bar()==i, NULL );
            vector_t v(dst_size, FooWithAssign(), init_alloc);
            for( int i=0; i<dst_size; ++i ) {
                ASSERT( v[i].state==Foo::CopyInitialized, NULL );
                v[i].bar() = ~i;
            }
            ASSERT( v != u, NULL);
            v.swap(u);
            CheckVector(u, dst_size, src_size);
            u.swap(v);
            // using assignment
            v = u;
            ASSERT( v == u, NULL);
            u.clear();
            ASSERT( u.size()==0, NULL );
            ASSERT( v.size()==size_t(src_size), NULL );
            for( int i=0; i<src_size; ++i )
                ASSERT( v[i].bar()==i, NULL );
            ASSERT( 0 == u.get_allocator().frees, NULL);
            u.shrink_to_fit(); // deallocate unused memory
            size_t items_allocated = u.get_allocator().items_allocated,
                   items_freed = u.get_allocator().items_freed;
            size_t allocations = u.get_allocator().allocations,
                   frees = u.get_allocator().frees + 100;
            ASSERT( items_allocated == items_freed, NULL);
            ASSERT( allocations == frees, NULL);
        }
    }
}

struct c_vector_type : default_container_traits {
    template<typename element_type, typename allocator_type>
    struct apply{
        typedef tbb::concurrent_vector<element_type,  allocator_type > type;
    };

    typedef FooIterator init_iterator_type;
    enum{ expected_number_of_items_to_allocate_for_steal_move = 0 };

    template<typename element_type, typename allocator_type, typename iterator>
    static bool equal(tbb::concurrent_vector<element_type, allocator_type > const& c, iterator begin, iterator end){
        bool equal_sizes = (size_t)std::distance(begin, end) == c.size();
        return  equal_sizes && std::equal(c.begin(), c.end(), begin);
    }
};

#if __TBB_CPP11_RVALUE_REF_PRESENT
void TestSerialGrowByWithMoveIterators(){
    typedef default_stateful_fixture_make_helper<c_vector_type>::type fixture_t;
    typedef fixture_t::container_t vector_t;

    fixture_t fixture("TestSerialGrowByWithMoveIterators");

    vector_t dst(fixture.dst_allocator);
    dst.grow_by(std::make_move_iterator(fixture.source.begin()), std::make_move_iterator(fixture.source.end()));

    fixture.verify_content_deep_moved(dst);
}

#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
namespace test_move_in_shrink_to_fit_helpers {
    struct dummy : Harness::StateTrackable<>{
        int i;
        dummy(int an_i) __TBB_NOEXCEPT(true) : Harness::StateTrackable<>(0), i(an_i) {}
#if !__TBB_IMPLICIT_MOVE_PRESENT || __TBB_NOTHROW_MOVE_MEMBERS_IMPLICIT_GENERATION_BROKEN
        dummy(const dummy &src) __TBB_NOEXCEPT(true) : Harness::StateTrackable<>(src), i(src.i) {}
        dummy(dummy &&src) __TBB_NOEXCEPT(true) : Harness::StateTrackable<>(std::move(src)), i(src.i) {}

        dummy& operator=(dummy &&src) __TBB_NOEXCEPT(true) {
            Harness::StateTrackable<>::operator=(std::move(src));
            i = src.i;
            return *this;
        }

        //somehow magically this declaration make std::is_nothrow_move_constructible<pod>::value to works correctly on icc14+msvc2013
        ~dummy() __TBB_NOEXCEPT(true) {}
#endif //!__TBB_IMPLICIT_MOVE_PRESENT || __TBB_NOTHROW_MOVE_MEMBERS_IMPLICIT_GENERATION_BROKEN
        friend bool operator== (const dummy &lhs, const dummy &rhs){ return lhs.i == rhs.i; }
    };
}
void TestSerialMoveInShrinkToFit(){
    const char* test_name = "TestSerialMoveInShrinkToFit";
    REMARK("running %s \n", test_name);
    using test_move_in_shrink_to_fit_helpers::dummy;

    __TBB_STATIC_ASSERT(std::is_nothrow_move_constructible<dummy>::value,"incorrect test setup or broken configuration?");
    {
        dummy src(0);
        ASSERT_IN_TEST(is_state<Harness::StateTrackableBase::MoveInitialized>(dummy(std::move_if_noexcept(src))),"broken configuration ?", test_name);
    }
    static const size_t sequence_size = 15;
    typedef  tbb::concurrent_vector<dummy> c_vector_t;
    std::vector<dummy> source(sequence_size, 0);
    std::generate_n(source.begin(), source.size(), std::rand);

    c_vector_t c_vector;
    c_vector.reserve(1); //make it fragmented

    c_vector.assign(source.begin(), source.end());
    memory_locations c_vector_before_shrink(c_vector);
    c_vector.shrink_to_fit();

    ASSERT_IN_TEST(c_vector_before_shrink.content_location_changed(c_vector), "incorrect test setup? shrink_to_fit should cause moving elements to other memory locations while it is not", test_name);
    ASSERT_IN_TEST(all_of(c_vector, is_state_f<Harness::StateTrackableBase::MoveInitialized>()), "container did not move construct some elements?", test_name);
    ASSERT_IN_TEST(c_vector == c_vector_t(source.begin(),source.end()),"",test_name);
}
#endif //__TBB_MOVE_IF_NOEXCEPT_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

#include <string>

// Test the comparison operators
void TestComparison() {
    std::string str[3]; str[0] = "abc";
    str[1].assign("cba");
    str[2].assign("abc"); // same as 0th
    tbb::concurrent_vector<char> var[3];
    var[0].assign(str[0].begin(), str[0].end());
    var[1].assign(str[0].rbegin(), str[0].rend());
    var[2].assign(var[1].rbegin(), var[1].rend()); // same as 0th
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT( (var[i] == var[j]) == (str[i] == str[j]), NULL );
            ASSERT( (var[i] != var[j]) == (str[i] != str[j]), NULL );
            ASSERT( (var[i] < var[j]) == (str[i] < str[j]), NULL );
            ASSERT( (var[i] > var[j]) == (str[i] > str[j]), NULL );
            ASSERT( (var[i] <= var[j]) == (str[i] <= str[j]), NULL );
            ASSERT( (var[i] >= var[j]) == (str[i] >= str[j]), NULL );
        }
    }
}

//------------------------------------------------------------------------
// Regression test for problem where on oversubscription caused
// concurrent_vector::grow_by to run very slowly (TR#196).
//------------------------------------------------------------------------

#include "tbb/task_scheduler_init.h"
#include <math.h>

typedef unsigned long Number;

static tbb::concurrent_vector<Number> Primes;

class FindPrimes {
    bool is_prime( Number val ) const {
        int limit, factor = 3;
        if( val<5u )
            return val==2;
        else {
            limit = long(sqrtf(float(val))+0.5f);
            while( factor<=limit && val % factor )
                ++factor;
            return factor>limit;
        }
    }
public:
    void operator()( const tbb::blocked_range<Number>& r ) const {
        for( Number i=r.begin(); i!=r.end(); ++i ) {
            if( i%2 && is_prime(i) ) {
                Primes.push_back( i );
            }
        }
    }
};

double TimeFindPrimes( int nthread ) {
    Primes.clear();
    Primes.reserve(1000000);// TODO: or compact()?
    tbb::task_scheduler_init init(nthread);
    tbb::tick_count t0 = tbb::tick_count::now();
    tbb::parallel_for( tbb::blocked_range<Number>(0,1000000,500), FindPrimes() );
    tbb::tick_count t1 = tbb::tick_count::now();
    return (t1-t0).seconds();
}

void TestFindPrimes() {
    // Time fully subscribed run.
    double t2 = TimeFindPrimes( tbb::task_scheduler_init::automatic );

    // Time parallel run that is very likely oversubscribed.
    double t128 = TimeFindPrimes(128);
    REMARK("TestFindPrimes: t2==%g t128=%g k=%g\n", t2, t128, t128/t2);

    // We allow the 128-thread run a little extra time to allow for thread overhead.
    // Theoretically, following test will fail on machine with >128 processors.
    // But that situation is not going to come up in the near future,
    // and the generalization to fix the issue is not worth the trouble.
    if( t128 > 1.3*t2 ) {
        REPORT("Warning: grow_by is pathetically slow: t2==%g t128=%g k=%g\n", t2, t128, t128/t2);
    }
}

//------------------------------------------------------------------------
// Test compatibility with STL sort.
//------------------------------------------------------------------------

#include <algorithm>

void TestSort() {
    for( int n=0; n<100; n=n*3+1 ) {
        tbb::concurrent_vector<int> array(n);
        for( int i=0; i<n; ++i )
            array.at(i) = (i*7)%n;
        std::sort( array.begin(), array.end() );
        for( int i=0; i<n; ++i )
            ASSERT( array[i]==i, NULL );
    }
}

#if TBB_USE_EXCEPTIONS

template<typename c_vector>
size_t get_early_size(c_vector & v){
      return v.grow_by(0) - v.begin();
}

void verify_c_vector_size(size_t size, size_t capacity, size_t early_size, const char * const test_name){
    ASSERT_IN_TEST( size <= capacity, "", test_name);
    ASSERT_IN_TEST( early_size >= size, "", test_name);
}

template<typename c_vector_t>
void verify_c_vector_size(c_vector_t & c_v, const char * const test_name){
    verify_c_vector_size(c_v.size(), c_v.capacity(), get_early_size(c_v), test_name);
}

void verify_c_vector_capacity_is_below(size_t capacity, size_t high, const char * const test_name){
    ASSERT_IN_TEST(capacity > 0, "unexpected capacity", test_name);
    ASSERT_IN_TEST(capacity < high, "unexpected capacity", test_name);
}

template<typename vector_t>
void verify_last_segment_allocation_failed(vector_t const& victim, const char* const test_name){
    ASSERT_THROWS_IN_TEST(victim.at(victim.size()), std::range_error, "",test_name );
}

template<typename vector_t>
void verify_assignment_operator_throws_bad_last_alloc(vector_t & victim, const char* const test_name){
    vector_t copy_of_victim(victim, victim.get_allocator());
    ASSERT_THROWS_IN_TEST(victim = copy_of_victim, tbb::bad_last_alloc, "", test_name);
}

template<typename vector_t>
void verify_copy_and_assign_from_produce_the_same(vector_t const& victim, const char* const test_name){
    //TODO: remove explicit copy of allocator when full support of C++11 allocator_traits in concurrent_vector is present
    vector_t copy_of_victim(victim, victim.get_allocator());
    ASSERT_IN_TEST(copy_of_victim == victim, "copy doesn't match original", test_name);
    vector_t copy_of_victim2(10, victim[0], victim.get_allocator());
    copy_of_victim2 = victim;
    ASSERT_IN_TEST(copy_of_victim == copy_of_victim2, "assignment doesn't match copying", test_name);
}

template<typename allocator_t>
void verify_vector_partially_copied(
        tbb::concurrent_vector<FooWithAssign, allocator_t> const& victim, size_t planned_victim_size,
        tbb::concurrent_vector<FooWithAssign, allocator_t> const& src,  bool is_memory_allocation_failure ,const char* const test_name)
{
    if (is_memory_allocation_failure) { // allocator generated exception
        typedef tbb::concurrent_vector<FooWithAssign, allocator_t> vector_t;
        ASSERT_IN_TEST( victim == vector_t(src.begin(), src.begin() + victim.size(), src.get_allocator()), "failed to properly copy of source ?", test_name );
    }else{
        ASSERT_IN_TEST( std::equal(victim.begin(), victim.begin() + planned_victim_size, src.begin()), "failed to properly copy items before the exception?", test_name );
        ASSERT_IN_TEST( ::all_of( victim.begin() + planned_victim_size, victim.end(), is_state_f<Foo::ZeroInitialized>() ), "failed to zero-initialize items left not constructed after the exception?", test_name );
    }
}

//------------------------------------------------------------------------
// Test exceptions safety (from allocator and items constructors)
//------------------------------------------------------------------------
void TestExceptions() {
    typedef static_counting_allocator<debug_allocator<FooWithAssign>, std::size_t> allocator_t;
    typedef tbb::concurrent_vector<FooWithAssign, allocator_t> vector_t;

    enum methods {
        zero_method = 0,
        ctor_copy, ctor_size, assign_nt, assign_ir, reserve, compact,
        all_methods
    };
    ASSERT( !FooCount, NULL );

    try {
        vector_t src(FooIterator(0), FooIterator(N)); // original data

        for(int t = 0; t < 2; ++t) // exception type
        for(int m = zero_method+1; m < all_methods; ++m)
        {
            track_foo_count<__LINE__> check_all_foo_destroyed_on_exit("TestExceptions");
            track_allocator_memory<allocator_t> verify_no_leak_at_exit("TestExceptions");
            allocator_t::init_counters();
            if(t) MaxFooCount = FooCount + N/4;
            else allocator_t::set_limits(N/4);
            vector_t victim;
            try {
                switch(m) {
                case ctor_copy: {
                        vector_t acopy(src);
                    } break; // auto destruction after exception is checked by ~Foo
                case ctor_size: {
                        vector_t sized(N);
                    } break; // auto destruction after exception is checked by ~Foo
                // Do not test assignment constructor due to reusing of same methods as below
                case assign_nt: {
                        victim.assign(N, FooWithAssign());
                    } break;
                case assign_ir: {
                        victim.assign(FooIterator(0), FooIterator(N));
                    } break;
                case reserve: {
                        try {
                            victim.reserve(victim.max_size()+1);
                        } catch(std::length_error &) {
                        } catch(...) {
                            KNOWN_ISSUE("ERROR: unrecognized exception - known compiler issue\n");
                        }
                        victim.reserve(N);
                    } break;
                case compact: {
                        if(t) MaxFooCount = 0; else allocator_t::set_limits(); // reset limits
                        victim.reserve(2); victim = src; // fragmented assignment
                        if(t) MaxFooCount = FooCount + 10; else allocator_t::set_limits(1, false); // block any allocation, check NULL return from allocator
                        victim.shrink_to_fit(); // should start defragmenting first segment
                    } break;
                default:;
                }
                if(!t || m != reserve) ASSERT(false, "should throw an exception");
            } catch(std::bad_alloc &e) {
                allocator_t::set_limits(); MaxFooCount = 0;
                size_t capacity = victim.capacity();
                size_t size = victim.size();

                size_t req_size = get_early_size(victim);

                verify_c_vector_size(size, capacity, req_size, "TestExceptions");

                switch(m) {
                case reserve:
                    if(t) ASSERT(false, NULL);
                case assign_nt:
                case assign_ir:
                    if(!t) {
                        ASSERT(capacity < N/2, "unexpected capacity");
                        ASSERT(size == 0, "unexpected size");
                        break;
                    } else {
                        ASSERT(size == N, "unexpected size");
                        ASSERT(capacity >= N, "unexpected capacity");
                        int i;
                        for(i = 1; ; ++i)
                            if(!victim[i].zero_bar()) break;
                            else ASSERT(victim[i].bar() == (m == assign_ir? i : initial_value_of_bar), NULL);
                        for(; size_t(i) < size; ++i) ASSERT(!victim[i].zero_bar(), NULL);
                        ASSERT(size_t(i) == size, NULL);
                        break;
                    }
                case compact:
                    ASSERT(capacity > 0, "unexpected capacity");
                    ASSERT(victim == src, "shrink_to_fit() is broken");
                    break;

                default:; // nothing to check here
                }
                REMARK("Exception %d: %s\t- ok\n", m, e.what());
            }
        }
    } catch(...) {
        ASSERT(false, "unexpected exception");
    }
}

//TODO: split into two separate tests
//TODO: remove code duplication in exception safety tests
void TestExceptionSafetyGuaranteesForAssignOperator(){
    //TODO: use __FUNCTION__ for test name
    const char* const test_name = "TestExceptionSafetyGuaranteesForAssignOperator";
    typedef static_counting_allocator<debug_allocator<FooWithAssign>, std::size_t> allocator_t;
    typedef tbb::concurrent_vector<FooWithAssign, allocator_t> vector_t;

    track_foo_count<__LINE__> check_all_foo_destroyed_on_exit(test_name);
    track_allocator_memory<allocator_t> verify_no_leak_at_exit(test_name);

    vector_t src(FooIterator(0), FooIterator(N)); // original data

    const size_t planned_victim_size = N/4;

    for(int t = 0; t < 2; ++t) {// exception type
        vector_t victim;
        victim.reserve(2); // get fragmented assignment

        ASSERT_THROWS_IN_TEST(
            {
                limit_foo_count_in_scope foo_limit(FooCount + planned_victim_size, t);
                limit_allocated_items_in_scope<allocator_t> allocator_limit(allocator_t::items_allocated + planned_victim_size, !t);

                victim = src; // fragmented assignment
            },
            std::bad_alloc, "", test_name
        );

        verify_c_vector_size(victim, test_name);

        if(!t) {
            verify_c_vector_capacity_is_below(victim.capacity(), N, test_name);
        }

        verify_vector_partially_copied(victim, planned_victim_size, src, !t, test_name);
        verify_last_segment_allocation_failed(victim, test_name);
        verify_copy_and_assign_from_produce_the_same(victim, test_name);
        verify_assignment_operator_throws_bad_last_alloc(victim, test_name);
    }
}
//TODO: split into two separate tests
void TestExceptionSafetyGuaranteesForConcurrentGrow(){
    const char* const test_name = "TestExceptionSafetyGuaranteesForConcurrentGrow";
    typedef static_counting_allocator<debug_allocator<FooWithAssign>, std::size_t> allocator_t;
    typedef tbb::concurrent_vector<FooWithAssign, allocator_t> vector_t;

    track_foo_count<__LINE__> check_all_foo_destroyed_on_exit(test_name);
    track_allocator_memory<allocator_t> verify_no_leak_at_exit(test_name);

    vector_t src(FooIterator(0), FooIterator(N)); // original data

    const size_t planned_victim_size = N/4;
    static const int grain_size = 70;

    tbb::task_scheduler_init init(2);

    for(int t = 0; t < 2; ++t) {// exception type
        vector_t victim;

#if TBB_USE_CAPTURED_EXCEPTION
        #define EXPECTED_EXCEPTION    tbb::captured_exception
#else
        #define EXPECTED_EXCEPTION    std::bad_alloc
#endif

        ASSERT_THROWS_IN_TEST(
            {
                limit_foo_count_in_scope foo_limit(FooCount +  31, t); // these numbers help to reproduce the live lock for versions < TBB2.2
                limit_allocated_items_in_scope<allocator_t> allocator_limit(allocator_t::items_allocated + planned_victim_size, !t);

                grain_map m(concurrent_grow_single_range_map, Harness::end(concurrent_grow_single_range_map));

                static const size_t part_weight =  grain_size / m.total_number_of_parts;

                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, N, grain_size),
                        GrowBy<vector_t>(victim, m, part_weight)
                );
            },
            EXPECTED_EXCEPTION, "", test_name
        );

        verify_c_vector_size(victim, test_name);

        if(!t) {
            verify_c_vector_capacity_is_below(victim.capacity(), N, test_name);
        }

        for(int i = 0; ; ++i) {
            try {
                Foo &foo = victim.at(i);
                ASSERT( foo.is_valid_or_zero(),"" );
            } catch(std::range_error &) { // skip broken segment
                ASSERT( size_t(i) < get_early_size(victim), NULL );
            } catch(std::out_of_range &){
                ASSERT( i > 0, NULL ); break;
            } catch(...) {
                KNOWN_ISSUE("ERROR: unrecognized exception - known compiler issue\n"); break;
            }
        }

        verify_copy_and_assign_from_produce_the_same(victim, test_name);
    }
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
void TestExceptionSafetyGuaranteesForMoveAssignOperatorWithUnEqualAllocatorMemoryFailure(){
    const char* const test_name = "TestExceptionSafetyGuaranteesForMoveAssignOperatorWithUnEqualAllocatorMemoryFailure";

    //TODO: add ability to inject debug_allocator into stateful_allocator_fixture::allocator_t
    //typedef static_counting_allocator<debug_allocator<FooWithAssign>, std::size_t> allocator_t;
    typedef default_stateful_fixture_make_helper<c_vector_type, Harness::false_type>::type fixture_t;
    typedef arena_allocator_fixture<FooWithAssign, Harness::false_type> arena_allocator_fixture_t;
    typedef fixture_t::allocator_t allocator_t;
    typedef fixture_t::container_t vector_t;

    fixture_t fixture(test_name);
    arena_allocator_fixture_t arena_allocator_fixture(4 * fixture.container_size);

    const size_t allocation_limit = fixture.container_size/4;

    vector_t victim(arena_allocator_fixture.allocator);
    victim.reserve(2); // get fragmented assignment

    ASSERT_THROWS_IN_TEST(
        {
            limit_allocated_items_in_scope<allocator_t> allocator_limit(allocator_t::items_allocated + allocation_limit);
            victim = std::move(fixture.source); // fragmented assignment
        },
        std::bad_alloc, "", test_name
    );

    verify_c_vector_size(victim, test_name);
    verify_c_vector_capacity_is_below(victim.capacity(), allocation_limit + 2, test_name);

    fixture.verify_part_of_content_deep_moved(victim, victim.size());

    verify_last_segment_allocation_failed(victim, test_name);
    verify_copy_and_assign_from_produce_the_same(victim, test_name);
    verify_assignment_operator_throws_bad_last_alloc(victim, test_name);
}

void TestExceptionSafetyGuaranteesForMoveAssignOperatorWithUnEqualAllocatorExceptionInElementCtor(){
    const char* const test_name = "TestExceptionSafetyGuaranteesForMoveAssignOperator";
    //typedef static_counting_allocator<debug_allocator<FooWithAssign>, std::size_t> allocator_t;
    typedef default_stateful_fixture_make_helper<c_vector_type, Harness::false_type>::type fixture_t;
    typedef arena_allocator_fixture<FooWithAssign, Harness::false_type> arena_allocator_fixture_t;
    typedef fixture_t::container_t vector_t;

    fixture_t fixture(test_name);
    const size_t planned_victim_size = fixture.container_size/4;
    arena_allocator_fixture_t arena_allocator_fixture(4 * fixture.container_size);

    vector_t victim(arena_allocator_fixture.allocator);
    victim.reserve(2); // get fragmented assignment

    ASSERT_THROWS_IN_TEST(
        {
            limit_foo_count_in_scope foo_limit(FooCount + planned_victim_size);
            victim = std::move(fixture.source); // fragmented assignment
        },
        std::bad_alloc, "", test_name
    );

    verify_c_vector_size(victim, test_name);

    fixture.verify_part_of_content_deep_moved(victim, planned_victim_size);

    verify_last_segment_allocation_failed(victim, test_name);
    verify_copy_and_assign_from_produce_the_same(victim, test_name);
    verify_assignment_operator_throws_bad_last_alloc(victim, test_name);
}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

namespace push_back_exception_safety_helpers{
    //TODO: remove code duplication with emplace_helpers::wrapper_type
    struct throwing_foo:Foo{
        int value1;
        int value2;
        explicit throwing_foo(int v1, int v2) : value1 (v1), value2(v2) {        }
    };

    template< typename foo_t = throwing_foo>
    struct fixture{
        typedef tbb::concurrent_vector<foo_t, debug_allocator<foo_t> > vector_t;
        vector_t v;

        void test( void(*p_test)(vector_t&), const char * test_name){
            track_foo_count<__LINE__> verify_no_foo_leaked_during_exception(test_name);
            ASSERT_IN_TEST(v.empty(),"incorrect test setup?", test_name );
            ASSERT_THROWS_IN_TEST(p_test(v), Foo_exception ,"", test_name);
            ASSERT_IN_TEST(is_state<Foo::ZeroInitialized>(v[0]),"incorrectly filled item during exception in emplace_back?", test_name);
        }
    };
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
void TestPushBackMoveExceptionSafety(){
    typedef push_back_exception_safety_helpers::fixture<Foo> fixture_t;
    fixture_t t;

    limit_foo_count_in_scope foo_limit(FooCount + 1);

    struct test{
        static void test_move_push_back(fixture_t::vector_t& v){
            Foo f;
            v.push_back(std::move(f));
        }
    };
    t.test(&test::test_move_push_back, "TestPushBackMoveExceptionSafety");
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
void TestEmplaceBackExceptionSafety(){
    typedef push_back_exception_safety_helpers::fixture<> fixture_t;
    fixture_t t;

    Foo dummy; //make FooCount non zero;
    Harness::suppress_unused_warning(dummy);
    limit_foo_count_in_scope foo_limit(FooCount);

    struct test{
        static void test_emplace(fixture_t::vector_t& v){
            v.emplace_back(1,2);
        }
    };
    t.test(&test::test_emplace, "TestEmplaceBackExceptionSafety");
}
#endif //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

#endif /* TBB_USE_EXCEPTIONS */

//------------------------------------------------------------------------
// Test support for SIMD instructions
//------------------------------------------------------------------------
#include "harness_m128.h"

#if HAVE_m128 || HAVE_m256

template<typename ClassWithVectorType>
void TestVectorTypes() {
    tbb::concurrent_vector<ClassWithVectorType> v;
    for( int i=0; i<100; ++i ) {
        // VC8 does not properly align a temporary value; to work around, use explicit variable
        ClassWithVectorType foo(i);
        v.push_back(foo);
        for( int j=0; j<i; ++j ) {
            ClassWithVectorType bar(j);
            ASSERT( v[j]==bar, NULL );
        }
    }
}
#endif /* HAVE_m128 | HAVE_m256 */

//------------------------------------------------------------------------

namespace v3_backward_compatibility{
    namespace segment_t_layout_helpers{
        //this is previous definition of according inner class of concurrent_vector_base_v3
        struct segment_t_v3 {
            void* array;
        };
        //helper class to access protected members of concurrent_vector_base
        struct access_vector_fields :tbb::internal::concurrent_vector_base_v3 {
            using tbb::internal::concurrent_vector_base_v3::segment_t;
            using tbb::internal::concurrent_vector_base_v3::segment_index_t;
            using tbb::internal::concurrent_vector_base_v3::pointers_per_long_table;
            using tbb::internal::concurrent_vector_base_v3::internal_segments_table;
        };
        //this is previous definition of according inner class of concurrent_vector_base_v3
        struct internal_segments_table_v3 {
            access_vector_fields::segment_index_t first_block;
            segment_t_v3 table[access_vector_fields::pointers_per_long_table];
        };

        template <typename checked_type>
        struct alignment_check_helper{
            char dummy;
            checked_type checked;
        };
    }
    void TestSegmentTLayout(){
        using namespace segment_t_layout_helpers;
        typedef alignment_check_helper<segment_t_v3> structure_with_old_segment_type;
        typedef alignment_check_helper<access_vector_fields::segment_t> structure_with_new_segment_type;

        ASSERT((sizeof(structure_with_old_segment_type)==sizeof(structure_with_new_segment_type))
              ,"layout of new segment_t and old one differ?");
    }

    void TestInternalSegmentsTableLayout(){
        using namespace segment_t_layout_helpers;
        typedef alignment_check_helper<internal_segments_table_v3> structure_with_old_segment_table_type;
        typedef alignment_check_helper<access_vector_fields::internal_segments_table> structure_with_new_segment_table_type;

        ASSERT((sizeof(structure_with_old_segment_table_type)==sizeof(structure_with_new_segment_table_type))
              ,"layout of new internal_segments_table and old one differ?");
    }
}
void TestV3BackwardCompatibility(){
    using namespace v3_backward_compatibility;
    TestSegmentTLayout();
    TestInternalSegmentsTableLayout();
}

#include "harness_defs.h"

#include <vector>
#include <numeric>
#include <functional>

// The helper to run a test only when a default construction is present.
template <bool default_construction_present> struct do_default_construction_test {
    template<typename FuncType> void operator() ( FuncType func ) const { func(); }
};
template <> struct do_default_construction_test<false> {
    template<typename FuncType> void operator()( FuncType ) const {}
};

template <typename Type, typename Allocator>
class test_grow_by_and_resize : NoAssign {
    tbb::concurrent_vector<Type, Allocator> &my_c;
public:
    test_grow_by_and_resize( tbb::concurrent_vector<Type, Allocator> &c ) : my_c(c) {}
    void operator()() const {
        const typename tbb::concurrent_vector<Type, Allocator>::size_type sz = my_c.size();
        my_c.grow_by( 5 );
        ASSERT( my_c.size() == sz + 5, NULL );
        my_c.resize( sz );
        ASSERT( my_c.size() == sz, NULL );
    }
};

template <typename Type, typename Allocator>
void CompareVectors( const tbb::concurrent_vector<Type, Allocator> &c1, const tbb::concurrent_vector<Type, Allocator> &c2 ) {
    ASSERT( !(c1 == c2) && c1 != c2, NULL );
    ASSERT( c1 <= c2 && c1 < c2 && c2 >= c1 && c2 > c1, NULL );
}

#if __TBB_CPP11_SMART_POINTERS_PRESENT
template <typename Type, typename Allocator>
void CompareVectors( const tbb::concurrent_vector<std::weak_ptr<Type>, Allocator> &, const tbb::concurrent_vector<std::weak_ptr<Type>, Allocator> & ) {
    /* do nothing for std::weak_ptr */
}
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */

template <bool default_construction_present, typename Type, typename Allocator>
void Examine( tbb::concurrent_vector<Type, Allocator> c, const std::vector<Type> &vec ) {
    typedef tbb::concurrent_vector<Type, Allocator> vector_t;
    typedef typename vector_t::size_type size_type_t;

    ASSERT( c.size() == vec.size(), NULL );
    for ( size_type_t i=0; i<c.size(); ++i ) ASSERT( Harness::IsEqual()(c[i], vec[i]), NULL );
    do_default_construction_test<default_construction_present>()(test_grow_by_and_resize<Type,Allocator>(c));
    c.grow_by( size_type_t(5), c[0] );
    c.grow_to_at_least( c.size()+5, c.at(0) );
    vector_t c2;
    c2.reserve( 5 );
    std::copy( c.begin(), c.begin() + 5, std::back_inserter( c2 ) );

    c.grow_by( c2.begin(), c2.end() );
    const vector_t& cvcr = c;
    ASSERT( Harness::IsEqual()(cvcr.front(), *(c2.rend()-1)), NULL );
    ASSERT( Harness::IsEqual()(cvcr.back(), *c2.rbegin()), NULL);
    ASSERT( Harness::IsEqual()(*c.cbegin(), *(c.crend()-1)), NULL );
    ASSERT( Harness::IsEqual()(*(c.cend()-1), *c.crbegin()), NULL );
    c.swap( c2 );
    ASSERT( c.size() == 5, NULL );
    CompareVectors( c, c2 );
    c.swap( c2 );
    c2.clear();
    ASSERT( c2.size() == 0, NULL );
    c2.shrink_to_fit();
    Allocator a = c.get_allocator();
    a.deallocate( a.allocate(1), 1 );
}

template <typename Type>
class test_default_construction : NoAssign {
    const std::vector<Type> &my_vec;
public:
    test_default_construction( const std::vector<Type> &vec ) : my_vec(vec) {}
    void operator()() const {
        // Construction with initial size specified by argument n.
        tbb::concurrent_vector<Type> c7( my_vec.size() );
        std::copy( my_vec.begin(), my_vec.end(), c7.begin() );
        Examine</*default_construction_present = */true>( c7, my_vec );
        tbb::concurrent_vector< Type, debug_allocator<Type> > c8( my_vec.size() );
        std::copy( c7.begin(), c7.end(), c8.begin() );
        Examine</*default_construction_present = */true>( c8, my_vec );
    }
};

template <bool default_construction_present, typename Type>
void TypeTester( const std::vector<Type> &vec ) {
    __TBB_ASSERT( vec.size() >= 5, "Array should have at least 5 elements" );
    // Construct empty vector.
    tbb::concurrent_vector<Type> c1;
    std::copy( vec.begin(), vec.end(), std::back_inserter(c1) );
    Examine<default_construction_present>( c1, vec );
#if __TBB_INITIALIZER_LISTS_PRESENT
    // Constructor from initializer_list.
    tbb::concurrent_vector<Type> c2({vec[0],vec[1],vec[2]});
    std::copy( vec.begin()+3, vec.end(), std::back_inserter(c2) );
    Examine<default_construction_present>( c2, vec );
#endif
    // Copying constructor.
    tbb::concurrent_vector<Type> c3(c1);
    Examine<default_construction_present>( c3, vec );
    // Construct with non-default allocator
    tbb::concurrent_vector< Type, debug_allocator<Type> > c4;
    std::copy( vec.begin(), vec.end(), std::back_inserter(c4) );
    Examine<default_construction_present>( c4, vec );
    // Copying constructor for vector with different allocator type.
    tbb::concurrent_vector<Type> c5(c4);
    Examine<default_construction_present>( c5, vec );
    tbb::concurrent_vector< Type, debug_allocator<Type> > c6(c3);
    Examine<default_construction_present>( c6, vec );
    // Construction with initial size specified by argument n.
    do_default_construction_test<default_construction_present>()(test_default_construction<Type>(vec));
    // Construction with initial size specified by argument n, initialization by copying of t, and given allocator instance.
    debug_allocator<Type> allocator;
    tbb::concurrent_vector< Type, debug_allocator<Type> > c9(vec.size(), vec[1], allocator);
    Examine<default_construction_present>( c9, std::vector<Type>(vec.size(), vec[1]) );
    // Construction with copying iteration range and given allocator instance.
    tbb::concurrent_vector< Type, debug_allocator<Type> > c10(c1.begin(), c1.end(), allocator);
    Examine<default_construction_present>( c10, vec );
    tbb::concurrent_vector<Type> c11(vec.begin(), vec.end());
    Examine<default_construction_present>( c11, vec );
}

void TestTypes() {
    const int NUMBER = 100;

    std::vector<int> intArr;
    for ( int i=0; i<NUMBER; ++i ) intArr.push_back(i);
    TypeTester</*default_construction_present = */true>( intArr );

#if __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN
    std::vector< std::reference_wrapper<int> > refArr;
    // The constructor of std::reference_wrapper<T> from T& is explicit in some versions of libstdc++.
    for ( int i=0; i<NUMBER; ++i ) refArr.push_back( std::reference_wrapper<int>(intArr[i]) );
    TypeTester</*default_construction_present = */false>( refArr );
#else
    REPORT( "Known issue: C++11 reference wrapper tests are skipped.\n" );
#endif /* __TBB_CPP11_REFERENCE_WRAPPER_PRESENT && !__TBB_REFERENCE_WRAPPER_COMPILATION_BROKEN */

    std::vector< tbb::atomic<int> > tbbIntArr( NUMBER );
    for ( int i=0; i<NUMBER; ++i ) tbbIntArr[i] = i;
    TypeTester</*default_construction_present = */true>( tbbIntArr );

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    std::vector< std::shared_ptr<int> > shrPtrArr;
    for ( int i=0; i<NUMBER; ++i ) shrPtrArr.push_back( std::make_shared<int>(i) );
    TypeTester</*default_construction_present = */true>( shrPtrArr );

    std::vector< std::weak_ptr<int> > wkPtrArr;
    std::copy( shrPtrArr.begin(), shrPtrArr.end(), std::back_inserter(wkPtrArr) );
    TypeTester</*default_construction_present = */true>( wkPtrArr );
#else
    REPORT( "Known issue: C++11 smart pointer tests are skipped.\n" );
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}

int TestMain () {
    if( MinThread<1 ) {
        REPORT("ERROR: MinThread=%d, but must be at least 1\n",MinThread); MinThread = 1;
    }
    TestFoo();
    TestV3BackwardCompatibility();
    TestIteratorTraits<tbb::concurrent_vector<Foo>::iterator,Foo>();
    TestIteratorTraits<tbb::concurrent_vector<Foo>::const_iterator,const Foo>();
    TestArrayLength();
    TestAllOf();
#if __TBB_INITIALIZER_LISTS_PRESENT
    TestInitList();
#else
    REPORT("Known issue: initializer list tests are skipped.\n");
#endif
    TestSequentialFor<FooWithAssign> ();
    TestResizeAndCopy();
    TestAssign();
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveConstructor<c_vector_type>();
    TestMoveAssignOperator<c_vector_type>();
    TestConstructorWithMoveIterators<c_vector_type>();
    TestAssignWithMoveIterators<c_vector_type>();
    TestSerialGrowByWithMoveIterators();
#if __TBB_MOVE_IF_NOEXCEPT_PRESENT
    TestSerialMoveInShrinkToFit();
#endif // __TBB_MOVE_IF_NOEXCEPT_PRESENT
#else
    REPORT("Known issue: tests for vector move constructor/assignment operator are skipped.\n");
#endif
    TestGrowToAtLeastWithSourceParameter<tbb::concurrent_vector<int> >(12345);
    TestSerialGrowByRange(false);
    TestSerialGrowByRange(true);
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestPushBackMoveOnlyContainee();
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    TestEmplaceBack();
#endif  //__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
#endif  //__TBB_CPP11_RVALUE_REF_PRESENT
#if HAVE_m128
    TestVectorTypes<ClassWithSSE>();
#endif
#if HAVE_m256
    if (have_AVX()) TestVectorTypes<ClassWithAVX>();
#endif
    TestCapacity();
    ASSERT( !FooCount, NULL );
    for( int nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        tbb::task_scheduler_init init( nthread );
        TestParallelFor( nthread );
        TestConcurrentGrowToAtLeast();
        TestConcurrentGrowBy( nthread );
    }
    ASSERT( !FooCount, NULL );
    TestComparison();
    TestFindPrimes();
    TestSort();
#if __TBB_RANGE_BASED_FOR_PRESENT
    TestRangeBasedFor();
#endif //if __TBB_RANGE_BASED_FOR_PRESENT
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception safety test is skipped.\n");
#elif TBB_USE_EXCEPTIONS
    TestExceptions();
    TestExceptionSafetyGuaranteesForAssignOperator();
#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorMemoryFailure<c_vector_type>();
    TestExceptionSafetyGuaranteesMoveConstructorWithUnEqualAllocatorExceptionInElementCtor<c_vector_type>();
    TestExceptionSafetyGuaranteesForMoveAssignOperatorWithUnEqualAllocatorMemoryFailure();
    TestExceptionSafetyGuaranteesForMoveAssignOperatorWithUnEqualAllocatorExceptionInElementCtor();
    TestPushBackMoveExceptionSafety();
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    TestEmplaceBackExceptionSafety();
#endif /*__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#else
    REPORT("Known issue: exception safety tests for move constructor/assignment operator , grow_by are skipped.\n");
#endif /*__TBB_CPP11_RVALUE_REF_PRESENT */
#endif /* TBB_USE_EXCEPTIONS */
    TestTypes();
    ASSERT( !FooCount, NULL );
    REMARK("sizeof(concurrent_vector<int>) == %d\n", (int)sizeof(tbb::concurrent_vector<int>));
    return Harness::Done;
}

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4800 is back
