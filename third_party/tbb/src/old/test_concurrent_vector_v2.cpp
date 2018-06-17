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

#include "concurrent_vector_v2.h"
#include <cstdio>
#include <cstdlib>
#include "../test/harness_assert.h"

tbb::atomic<long> FooCount;

//! Problem size
const size_t N = 500000;

struct Foo {
    int my_bar;
public:
    enum State {
        DefaultInitialized=0x1234,
        CopyInitialized=0x89ab,
        Destroyed=0x5678
    } state;
    int& bar() {
        ASSERT( state==DefaultInitialized||state==CopyInitialized, NULL );
        return my_bar;
    }
    int bar() const {
        ASSERT( state==DefaultInitialized||state==CopyInitialized, NULL );
        return my_bar;
    }
    static const int initial_value_of_bar = 42;
    Foo() {
        state = DefaultInitialized;
        ++FooCount;
        my_bar = initial_value_of_bar;
    }
    Foo( const Foo& foo ) {
        state = CopyInitialized;
        ++FooCount;
        my_bar = foo.my_bar;
    }
    ~Foo() {
        ASSERT( state==DefaultInitialized||state==CopyInitialized, NULL );
        state = Destroyed;
        my_bar = ~initial_value_of_bar;
        --FooCount;
    }
    bool is_const() const {return true;}
    bool is_const() {return false;}
};

class FooWithAssign: public Foo {
public:
    void operator=( const FooWithAssign& x ) {
        ASSERT( x.state==DefaultInitialized||x.state==CopyInitialized, NULL );
        ASSERT( state==DefaultInitialized||state==CopyInitialized, NULL );
        my_bar = x.my_bar;
    }
};

inline void NextSize( int& s ) {
    if( s<=32 ) ++s;
    else s += s/10;
}

static void CheckVector( const tbb::concurrent_vector<Foo>& cv, size_t expected_size, size_t old_size ) {
    ASSERT( cv.size()==expected_size, NULL );
    ASSERT( cv.empty()==(expected_size==0), NULL );
    for( int j=0; j<int(expected_size); ++j ) {
        if( cv[j].bar()!=~j )
            std::printf("ERROR on line %d for old_size=%ld expected_size=%ld j=%d\n",__LINE__,long(old_size),long(expected_size),j);
    }
}

void TestResizeAndCopy() {
    typedef tbb::concurrent_vector<Foo> vector_t;
    for( int old_size=0; old_size<=128; NextSize( old_size ) ) {
        for( int new_size=old_size; new_size<=128; NextSize( new_size ) ) {
            long count = FooCount;
            vector_t v;
            ASSERT( count==FooCount, NULL );
            v.grow_by(old_size);
            ASSERT( count+old_size==FooCount, NULL );
            for( int j=0; j<old_size; ++j )
                v[j].bar() = j*j;
            v.grow_to_at_least(new_size);
            ASSERT( count+new_size==FooCount, NULL );
            for( int j=0; j<new_size; ++j ) {
                int expected = j<old_size ? j*j : Foo::initial_value_of_bar;
                if( v[j].bar()!=expected )
                    std::printf("ERROR on line %d for old_size=%ld new_size=%ld v[%ld].bar()=%d != %d\n",__LINE__,long(old_size),long(new_size),long(j),v[j].bar(), expected);
            }
            ASSERT( v.size()==size_t(new_size), NULL );
            for( int j=0; j<new_size; ++j ) {
                v[j].bar() = ~j;
            }
            const vector_t& cv = v;
            // Try copy constructor
            vector_t copy_of_v(cv);
            CheckVector(cv,new_size,old_size);
            v.clear();
            ASSERT( v.empty(), NULL );
            CheckVector(copy_of_v,new_size,old_size);
        }
    }
}

void TestCapacity() {
    for( size_t old_size=0; old_size<=10000; old_size=(old_size<5 ? old_size+1 : 3*old_size) ) {
        for( size_t new_size=0; new_size<=10000; new_size=(new_size<5 ? new_size+1 : 3*new_size) ) {
            long count = FooCount;
            {
                typedef tbb::concurrent_vector<Foo> vector_t;
                vector_t v;
                v.reserve( old_size );
                ASSERT( v.capacity()>=old_size, NULL );
                v.reserve( new_size );
                ASSERT( v.capacity()>=old_size, NULL );
                ASSERT( v.capacity()>=new_size, NULL );
                for( size_t i=0; i<2*new_size; ++i ) {
                    ASSERT( size_t(FooCount)==count+i, NULL );
                    size_t j = v.grow_by(1);
                    ASSERT( j==i, NULL );
                }
            }
            ASSERT( FooCount==count, NULL );
        }
    }
}

struct AssignElement {
    typedef tbb::concurrent_vector<int>::range_type::iterator iterator;
    iterator base;
    void operator()( const tbb::concurrent_vector<int>::range_type& range ) const {
        for( iterator i=range.begin(); i!=range.end(); ++i ) {
            if( *i!=0 )
                std::printf("ERROR for v[%ld]\n", long(i-base));
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
                std::printf("ERROR for v[%ld]\n", long(i-base));
    }
    CheckElement( iterator base_ ) : base(base_) {}
};

#include "tbb/tick_count.h"
#include "tbb/parallel_for.h"
#include "../test/harness.h"

//! Test parallel access by iterators
void TestParallelFor( int nthread ) {
    typedef tbb::concurrent_vector<int> vector_t;
    vector_t v;
    v.grow_to_at_least(N);
    tbb::tick_count t0 = tbb::tick_count::now();
    if( Verbose )
        std::printf("Calling parallel_for.h with %ld threads\n",long(nthread));
    tbb::parallel_for( v.range(10000), AssignElement(v.begin()) );
    tbb::tick_count t1 = tbb::tick_count::now();
    const vector_t& u = v;
    tbb::parallel_for( u.range(10000), CheckElement(u.begin()) );
    tbb::tick_count t2 = tbb::tick_count::now();
    if( Verbose )
        std::printf("Time for parallel_for.h: assign time = %8.5f, check time = %8.5f\n",
               (t1-t0).seconds(),(t2-t1).seconds());
    for( long i=0; size_t(i)<v.size(); ++i )
        if( v[i]!=i )
            std::printf("ERROR for v[%ld]\n", i);
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
        std::printf("ERROR for u[%ld] using const_iterator\n", long(i));
    typename Vector::difference_type delta = cp-u.begin();
    ASSERT( delta==i, NULL );
    if( u[i].bar()!=i )
        std::printf("ERROR for u[%ld] using subscripting\n", long(i));
    ASSERT( u.begin()[i].bar()==i, NULL );
}

template<typename Iterator1, typename Iterator2, typename V>
void CheckIteratorComparison( V& u ) {
    Iterator1 i = u.begin();
    for( int i_count=0; i_count<100; ++i_count ) {
        Iterator2 j = u.begin();
        for( int j_count=0; j_count<100; ++j_count ) {
            ASSERT( (i==j)==(i_count==j_count), NULL );
            ASSERT( (i!=j)==(i_count!=j_count), NULL );
            ASSERT( (i-j)==(i_count-j_count), NULL );
            ASSERT( (i<j)==(i_count<j_count), NULL );
            ASSERT( (i>j)==(i_count>j_count), NULL );
            ASSERT( (i<=j)==(i_count<=j_count), NULL );
            ASSERT( (i>=j)==(i_count>=j_count), NULL );
            ++j;
        }
        ++i;
    }
}

//! Test sequential iterators for vector type V.
/** Also does timing. */
template<typename V>
void TestSequentialFor() {
    V v;
    v.grow_by(N);

    // Check iterator
    tbb::tick_count t0 = tbb::tick_count::now();
    typename V::iterator p = v.begin();
    ASSERT( !(*p).is_const(), NULL );
    ASSERT( !p->is_const(), NULL );
    for( int i=0; size_t(i)<v.size(); ++i, ++p ) {
        if( (*p).state!=Foo::DefaultInitialized )
            std::printf("ERROR for v[%ld]\n", long(i));
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
    ASSERT( (*cp).is_const(), NULL );
    ASSERT( cp->is_const(), NULL );
    for( int i=0; size_t(i)<u.size(); ++i, ++cp ) {
        CheckConstIterator(u,i,cp);
    }
    tbb::tick_count t2 = tbb::tick_count::now();
    if( Verbose )
        std::printf("Time for serial for:  assign time = %8.5f, check time = %8.5f\n",
               (t1-t0).seconds(),(t2-t1).seconds());

    // Now go backwards
    cp = u.end();
    for( int i=int(u.size()); i>0; ) {
        --i;
        --cp;
        if( i>0 ) {
            typename V::const_iterator cp_old = cp--;
            int here = (*cp_old).bar();
            ASSERT( here==u[i].bar(), NULL );
            typename V::const_iterator cp_new = cp++;
            int prev = (*cp_new).bar();
            ASSERT( prev==u[i-1].bar(), NULL );
        }
        CheckConstIterator(u,i,cp);
    }

    // Now go forwards and backwards
    cp = u.begin();
    ptrdiff_t k = 0;
    for( size_t i=0; i<u.size(); ++i ) {
        CheckConstIterator(u,int(k),cp);
        typename V::difference_type delta = i*3 % u.size();
        if( 0<=k+delta && size_t(k+delta)<u.size() ) {
            cp += delta;
            k += delta;
        }
        delta = i*7 % u.size();
        if( 0<=k-delta && size_t(k-delta)<u.size() ) {
            if( i&1 )
                cp -= delta;            // Test operator-=
            else
                cp = cp - delta;        // Test operator-
            k -= delta;
        }
    }

    for( int i=0; size_t(i)<u.size(); i=(i<50?i+1:i*3) )
        for( int j=-i; size_t(i+j)<u.size(); j=(j<50?j+1:j*5) ) {
            ASSERT( (u.begin()+i)[j].bar()==i+j, NULL );
            ASSERT( (v.begin()+i)[j].bar()==i+j, NULL );
            ASSERT( (i+u.begin())[j].bar()==i+j, NULL );
            ASSERT( (i+v.begin())[j].bar()==i+j, NULL );
        }

    CheckIteratorComparison<typename V::iterator, typename V::iterator>(v);
    CheckIteratorComparison<typename V::iterator, typename V::const_iterator>(v);
    CheckIteratorComparison<typename V::const_iterator, typename V::iterator>(v);
    CheckIteratorComparison<typename V::const_iterator, typename V::const_iterator>(v);

    TestIteratorAssignment<typename V::const_iterator>( u.begin() );
    TestIteratorAssignment<typename V::const_iterator>( v.begin() );
    TestIteratorAssignment<typename V::iterator>( v.begin() );

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
    for( size_t i=v.size(); i>0; --i, ++crp ) {
        typename V::const_reference cpref = *crp;
        ASSERT( size_t(cpref.bar())==i-1, NULL );
        ASSERT( crp!=u.rend(), NULL );
    }
    ASSERT( crp==u.rend(), NULL );

    TestIteratorAssignment<typename V::const_reverse_iterator>( u.rbegin() );
    TestIteratorAssignment<typename V::reverse_iterator>( v.rbegin() );
}

static const size_t Modulus = 7;

typedef tbb::concurrent_vector<Foo> MyVector;

class GrowToAtLeast {
    MyVector& my_vector;
public:
    void operator()( const tbb::blocked_range<size_t>& range ) const {
        for( size_t i=range.begin(); i!=range.end(); ++i ) {
            size_t n = my_vector.size();
            size_t k = n==0 ? 0 : i % (2*n+1);
            my_vector.grow_to_at_least(k+1);
            ASSERT( my_vector.size()>=k+1, NULL );
        }
    }
    GrowToAtLeast( MyVector& vector ) : my_vector(vector) {}
};

void TestConcurrentGrowToAtLeast() {
    MyVector v;
    for( size_t s=1; s<1000; s*=10 ) {
        tbb::parallel_for( tbb::blocked_range<size_t>(0,1000000,100), GrowToAtLeast(v) );
    }
}

//! Test concurrent invocations of method concurrent_vector::grow_by
class GrowBy {
    MyVector& my_vector;
public:
    void operator()( const tbb::blocked_range<int>& range ) const {
        for( int i=range.begin(); i!=range.end(); ++i ) {
            if( i%3 ) {
                Foo& element = my_vector[my_vector.grow_by(1)];
                element.bar() = i;
            } else {
                Foo f;
                f.bar() = i;
                size_t k = my_vector.push_back( f );
                ASSERT( my_vector[k].bar()==i, NULL );
            }
        }
    }
    GrowBy( MyVector& vector ) : my_vector(vector) {}
};

//! Test concurrent invocations of method concurrent_vector::grow_by
void TestConcurrentGrowBy( int nthread ) {
    int m = 100000;
    MyVector v;
    tbb::parallel_for( tbb::blocked_range<int>(0,m,1000), GrowBy(v) );
    ASSERT( v.size()==size_t(m), NULL );

    // Verify that v is a permutation of 0..m
    int inversions = 0;
    bool* found = new bool[m];
    memset( found, 0, m );
    for( int i=0; i<m; ++i ) {
        int index = v[i].bar();
        ASSERT( !found[index], NULL );
        found[index] = true;
        if( i>0 )
            inversions += v[i].bar()<v[i-1].bar();
    }
    for( int i=0; i<m; ++i ) {
        ASSERT( found[i], NULL );
        ASSERT( nthread>1 || v[i].bar()==i, "sequential execution is wrong" );
    }
    delete[] found;
    if( nthread>1 && inversions<m/10 )
        std::printf("Warning: not much concurrency in TestConcurrentGrowBy\n");
}

//! Test the assignment operator
void TestAssign() {
    typedef tbb::concurrent_vector<FooWithAssign> vector_t;
    for( int dst_size=1; dst_size<=128; NextSize( dst_size ) ) {
        for( int src_size=2; src_size<=128; NextSize( src_size ) ) {
            vector_t u;
            u.grow_to_at_least(src_size);
            for( int i=0; i<src_size; ++i )
                u[i].bar() = i*i;
            vector_t v;
            v.grow_to_at_least(dst_size);
            for( int i=0; i<dst_size; ++i )
                v[i].bar() = -i;
            v = u;
            u.clear();
            ASSERT( u.size()==0, NULL );
            ASSERT( v.size()==size_t(src_size), NULL );
            for( int i=0; i<src_size; ++i )
                ASSERT( v[i].bar()==(i*i), NULL );
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
                Primes[Primes.grow_by(1)] = i;
            }
        }
    }
};

static double TimeFindPrimes( int nthread ) {
    Primes.clear();
    tbb::task_scheduler_init init(nthread);
    tbb::tick_count t0 = tbb::tick_count::now();
    tbb::parallel_for( tbb::blocked_range<Number>(0,1000000,500), FindPrimes() );
    tbb::tick_count t1 = tbb::tick_count::now();
    return (t1-t0).seconds();
}

static void TestFindPrimes() {
    // Time fully subscribed run.
    double t2 = TimeFindPrimes( tbb::task_scheduler_init::automatic );

    // Time parallel run that is very likely oversubscribed.
    double t128 = TimeFindPrimes(128);

    if( Verbose )
        std::printf("TestFindPrimes: t2==%g t128=%g\n", t2, t128 );

    // We allow the 128-thread run a little extra time to allow for thread overhead.
    // Theoretically, following test will fail on machine with >128 processors.
    // But that situation is not going to come up in the near future,
    // and the generalization to fix the issue is not worth the trouble.
    if( t128>1.10*t2 ) {
        std::printf("Warning: grow_by is pathetically slow: t2==%g t128=%g\n", t2, t128);
    }
}

//------------------------------------------------------------------------
// Test compatibility with STL sort.
//------------------------------------------------------------------------

#include <algorithm>

void TestSort() {
    for( int n=1; n<100; n*=3 ) {
        tbb::concurrent_vector<int> array;
        array.grow_by( n );
        for( int i=0; i<n; ++i )
            array[i] = (i*7)%n;
        std::sort( array.begin(), array.end() );
        for( int i=0; i<n; ++i )
            ASSERT( array[i]==i, NULL );
    }
}

//------------------------------------------------------------------------

int TestMain () {
    if( MinThread<1 ) {
        std::printf("ERROR: MinThread=%d, but must be at least 1\n",MinThread);
    }

    TestIteratorTraits<tbb::concurrent_vector<Foo>::iterator,Foo>();
    TestIteratorTraits<tbb::concurrent_vector<Foo>::const_iterator,const Foo>();
    TestSequentialFor<tbb::concurrent_vector<Foo> > ();
    TestResizeAndCopy();
    TestAssign();
    TestCapacity();
    for( int nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        tbb::task_scheduler_init init( nthread );
        TestParallelFor( nthread );
        TestConcurrentGrowToAtLeast();
        TestConcurrentGrowBy( nthread );
    }
    TestFindPrimes();
    TestSort();
    return Harness::Done;
}
