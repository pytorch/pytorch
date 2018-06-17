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

#define NOMINMAX
#include "harness_defs.h"
#include "tbb/concurrent_queue.h"
#include "tbb/tick_count.h"
#include "harness.h"
#include "harness_allocator.h"

using tbb::internal::spin_wait_while;

#include <vector>

static tbb::atomic<long> FooConstructed;
static tbb::atomic<long> FooDestroyed;

enum state_t{
    LIVE=0x1234,
    DEAD=0xDEAD
};

class Foo {
    state_t state;
public:
    int thread_id;
    int serial;
    Foo() : state(LIVE), thread_id(0), serial(0) {
        ++FooConstructed;
    }
    Foo( const Foo& item ) : state(LIVE) {
        ASSERT( item.state==LIVE, NULL );
        ++FooConstructed;
        thread_id = item.thread_id;
        serial = item.serial;
    }
    ~Foo() {
        ASSERT( state==LIVE, NULL );
        ++FooDestroyed;
        state=DEAD;
        thread_id=DEAD;
        serial=DEAD;
    }
    void operator=( const Foo& item ) {
        ASSERT( item.state==LIVE, NULL );
        ASSERT( state==LIVE, NULL );
        thread_id = item.thread_id;
        serial = item.serial;
    }
    bool is_const() {return false;}
    bool is_const() const {return true;}
    static void clear_counters() { FooConstructed = 0; FooDestroyed = 0; }
    static long get_n_constructed() { return FooConstructed; }
    static long get_n_destroyed() { return FooDestroyed; }
};

// problem size
static const int N = 50000;     // # of bytes

#if TBB_USE_EXCEPTIONS
//! Exception for concurrent_queue
class Foo_exception : public std::bad_alloc {
public:
    virtual const char *what() const throw() __TBB_override { return "out of Foo limit"; }
    virtual ~Foo_exception() throw() {}
};

static tbb::atomic<long> FooExConstructed;
static tbb::atomic<long> FooExDestroyed;
static tbb::atomic<long> serial_source;
static long MaxFooCount = 0;
static const long Threshold = 400;

class FooEx {
    state_t state;
public:
    int serial;
    FooEx() : state(LIVE) {
        ++FooExConstructed;
        serial = serial_source++;
    }
    FooEx( const FooEx& item ) : state(LIVE) {
        ASSERT( item.state == LIVE, NULL );
        ++FooExConstructed;
        if( MaxFooCount && (FooExConstructed-FooExDestroyed) >= MaxFooCount ) // in push()
            throw Foo_exception();
        serial = item.serial;
    }
    ~FooEx() {
        ASSERT( state==LIVE, NULL );
        ++FooExDestroyed;
        state=DEAD;
        serial=DEAD;
    }
    void operator=( FooEx& item ) {
        ASSERT( item.state==LIVE, NULL );
        ASSERT( state==LIVE, NULL );
        serial = item.serial;
        if( MaxFooCount==2*Threshold && (FooExConstructed-FooExDestroyed) <= MaxFooCount/4 ) // in pop()
            throw Foo_exception();
    }
#if __TBB_CPP11_RVALUE_REF_PRESENT
    void operator=( FooEx&& item ) {
        operator=( item );
        item.serial = 0;
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
} ;
#endif /* TBB_USE_EXCEPTIONS */

const size_t MAXTHREAD = 256;

static int Sum[MAXTHREAD];

//! Count of various pop operations
/** [0] = pop_if_present that failed
    [1] = pop_if_present that succeeded
    [2] = pop */
static tbb::atomic<long> PopKind[3];

const int M = 10000;

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT
const size_t push_selector_variants = 3;
#elif __TBB_CPP11_RVALUE_REF_PRESENT
const size_t push_selector_variants = 2;
#else
const size_t push_selector_variants = 1;
#endif

template<typename CQ, typename ValueType, typename CounterType>
void push( CQ& q, ValueType v, CounterType i ) {
    switch( i % push_selector_variants ) {
    case 0: q.push( v ); break;
#if __TBB_CPP11_RVALUE_REF_PRESENT
    case 1: q.push( std::move(v) ); break;
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    case 2: q.emplace( v ); break;
#endif
#endif
    default: ASSERT( false, NULL ); break;
    }
}

template<typename CQ,typename T>
struct Body: NoAssign {
    CQ* queue;
    const int nthread;
    Body( int nthread_ ) : nthread(nthread_) {}
    void operator()( int thread_id ) const {
        long pop_kind[3] = {0,0,0};
        int serial[MAXTHREAD+1];
        memset( serial, 0, nthread*sizeof(int) );
        ASSERT( thread_id<nthread, NULL );

        long sum = 0;
        for( long j=0; j<M; ++j ) {
            T f;
            f.thread_id = DEAD;
            f.serial = DEAD;
            bool prepopped = false;
            if( j&1 ) {
                prepopped = queue->try_pop( f );
                ++pop_kind[prepopped];
            }
            T g;
            g.thread_id = thread_id;
            g.serial = j+1;
            push( *queue, g, j );
            if( !prepopped ) {
                while( !(queue)->try_pop(f) ) __TBB_Yield();
                ++pop_kind[2];
            }
            ASSERT( f.thread_id<=nthread, NULL );
            ASSERT( f.thread_id==nthread || serial[f.thread_id]<f.serial, "partial order violation" );
            serial[f.thread_id] = f.serial;
            sum += f.serial-1;
        }
        Sum[thread_id] = sum;
        for( int k=0; k<3; ++k )
            PopKind[k] += pop_kind[k];
    }
};

// Define wrapper classes to test tbb::concurrent_queue<T>
template<typename T, typename A = tbb::cache_aligned_allocator<T> >
class ConcQWithSizeWrapper : public tbb::concurrent_queue<T, A> {
public:
    ConcQWithSizeWrapper() {}
    ConcQWithSizeWrapper( const ConcQWithSizeWrapper& q ) : tbb::concurrent_queue<T, A>( q ) {}
    ConcQWithSizeWrapper(const A& a) : tbb::concurrent_queue<T, A>( a ) {}
#if __TBB_CPP11_RVALUE_REF_PRESENT
    ConcQWithSizeWrapper(ConcQWithSizeWrapper&& q) : tbb::concurrent_queue<T>( std::move(q) ) {}
    ConcQWithSizeWrapper(ConcQWithSizeWrapper&& q, const A& a)
        : tbb::concurrent_queue<T, A>( std::move(q), a ) { }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
    template<typename InputIterator>
    ConcQWithSizeWrapper( InputIterator begin, InputIterator end, const A& a = A())
        : tbb::concurrent_queue<T, A>(begin,end,a) {}
    size_t size() const { return this->unsafe_size(); }
};

template<typename T>
class ConcQPushPopWrapper : public tbb::concurrent_queue<T> {
public:
    ConcQPushPopWrapper() : my_capacity( size_t(-1)/(sizeof(void*)+sizeof(T)) ) {}
    size_t size() const { return this->unsafe_size(); }
    void   set_capacity( const ptrdiff_t n ) { my_capacity = n; }
    bool   try_push( const T& source ) { return this->push( source ); }
    bool   try_pop( T& dest ) { return this->tbb::concurrent_queue<T>::try_pop( dest ); }
    size_t my_capacity;
};

template<typename T>
class ConcQWithCapacity : public tbb::concurrent_queue<T> {
public:
    ConcQWithCapacity() : my_capacity( size_t(-1)/(sizeof(void*)+sizeof(T)) ) {}
    size_t size() const { return this->unsafe_size(); }
    size_t capacity() const { return my_capacity; }
    void   set_capacity( const int n ) { my_capacity = n; }
    bool   try_push( const T& source ) { this->push( source ); return (size_t)source.serial<my_capacity; }
    bool   try_pop( T& dest ) { this->tbb::concurrent_queue<T>::try_pop( dest ); return (size_t)dest.serial<my_capacity; }
    size_t my_capacity;
};

template <typename Queue>
void AssertEquality(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    ASSERT(q.size() == typename Queue::size_type(vec.size()), NULL);
    ASSERT(std::equal(q.unsafe_begin(), q.unsafe_end(), vec.begin(), Harness::IsEqual()), NULL);
}

template <typename Queue>
void AssertEmptiness(Queue &q) {
    ASSERT(q.empty(), NULL);
    ASSERT(!q.size(), NULL);
    typename Queue::value_type elem;
    ASSERT(!q.try_pop(elem), NULL);
}

enum push_t { push_op, try_push_op };

template<push_t push_op>
struct pusher {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename CQ, typename VType>
    static bool push( CQ& queue, VType&& val ) {
        queue.push( std::forward<VType>( val ) );
        return true;
    }
#else
    template<typename CQ, typename VType>
    static bool push( CQ& queue, const VType& val ) {
        queue.push( val );
        return true;
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
};

template<>
struct pusher< try_push_op > {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename CQ, typename VType>
    static bool push( CQ& queue, VType&& val ) {
        return queue.try_push( std::forward<VType>( val ) );
    }
#else
    template<typename CQ, typename VType>
    static bool push( CQ& queue, const VType& val ) {
        return queue.try_push( val );
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
};

enum pop_t { pop_op, try_pop_op };

template<pop_t pop_op>
struct popper {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType&& val ) {
        if( queue.empty() ) return false;
        queue.pop( std::forward<VType>( val ) );
        return true;
    }
#else
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType& val ) {
        if( queue.empty() ) return false;
        queue.pop( val );
        return true;
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
};

template<>
struct popper< try_pop_op > {
#if __TBB_CPP11_RVALUE_REF_PRESENT
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType&& val ) {
        return queue.try_pop( std::forward<VType>( val ) );
    }
#else
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType& val ) {
        return queue.try_pop( val );
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
};

template <push_t push_op, typename Queue>
void FillTest(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    for (typename std::vector<typename Queue::value_type>::const_iterator it = vec.begin(); it != vec.end(); ++it)
        ASSERT(pusher<push_op>::push(q, *it), NULL);
    AssertEquality(q, vec);
}

template <pop_t pop_op, typename Queue>
void EmptyTest(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    typedef typename Queue::value_type value_type;

    value_type elem;
    typename std::vector<value_type>::const_iterator it = vec.begin();
    while (popper<pop_op>::pop(q, elem)) {
        ASSERT(Harness::IsEqual()(elem, *it), NULL);
        ++it;
    }
    ASSERT(it == vec.end(), NULL);
    AssertEmptiness(q);
}

template <typename T, typename A>
void bounded_queue_specific_test(tbb::concurrent_queue<T, A> &, const std::vector<T> &) { /* do nothing */ }

template <typename T, typename A>
void bounded_queue_specific_test(tbb::concurrent_bounded_queue<T, A> &q, const std::vector<T> &vec) {
    typedef typename tbb::concurrent_bounded_queue<T, A>::size_type size_type;

    FillTest<try_push_op>(q, vec);
    tbb::concurrent_bounded_queue<T, A> q2 = q;
    EmptyTest<pop_op>(q, vec);

    // capacity
    q2.set_capacity(size_type(vec.size()));
    ASSERT(q2.capacity() == size_type(vec.size()), NULL);
    ASSERT(q2.size() == size_type(vec.size()), NULL);
    ASSERT(!q2.try_push(vec[0]), NULL);

#if TBB_USE_EXCEPTIONS
    q.abort();
#endif
}

template<typename CQ, typename T>
void TestPushPop( size_t prefill, ptrdiff_t capacity, int nthread ) {
    ASSERT( nthread>0, "nthread must be positive" );
    ptrdiff_t signed_prefill = ptrdiff_t(prefill);
    if( signed_prefill+1>=capacity )
        return;
    bool success = false;
    for( int k=0; k<3; ++k )
        PopKind[k] = 0;
    for( int trial=0; !success; ++trial ) {
        T::clear_counters();
        Body<CQ,T> body(nthread);
        CQ queue;
        queue.set_capacity( capacity );
        body.queue = &queue;
        for( size_t i=0; i<prefill; ++i ) {
            T f;
            f.thread_id = nthread;
            f.serial = 1+int(i);
            push(queue, f, i);
            ASSERT( unsigned(queue.size())==i+1, NULL );
            ASSERT( !queue.empty(), NULL );
        }
        tbb::tick_count t0 = tbb::tick_count::now();
        NativeParallelFor( nthread, body );
        tbb::tick_count t1 = tbb::tick_count::now();
        double timing = (t1-t0).seconds();
        REMARK("prefill=%d capacity=%d threads=%d time = %g = %g nsec/operation\n", int(prefill), int(capacity), nthread, timing, timing/(2*M*nthread)*1.E9);
        int sum = 0;
        for( int k=0; k<nthread; ++k )
            sum += Sum[k];
        int expected = int(nthread*((M-1)*M/2) + ((prefill-1)*prefill)/2);
        for( int i=int(prefill); --i>=0; ) {
            ASSERT( !queue.empty(), NULL );
            T f;
            bool result = queue.try_pop(f);
            ASSERT( result, NULL );
            ASSERT( int(queue.size())==i, NULL );
            sum += f.serial-1;
        }
        ASSERT( queue.empty(), "The queue should be empty" );
        ASSERT( queue.size()==0, "The queue should have zero size" );
        if( sum!=expected )
            REPORT("sum=%d expected=%d\n",sum,expected);
        ASSERT( T::get_n_constructed()==T::get_n_destroyed(), NULL );
        // TODO: checks by counting allocators

        success = true;
        if( nthread>1 && prefill==0 ) {
            // Check that pop_if_present got sufficient exercise
            for( int k=0; k<2; ++k ) {
#if (_WIN32||_WIN64)
                // The TBB library on Windows seems to have a tough time generating
                // the desired interleavings for pop_if_present, so the code tries longer, and settles
                // for fewer desired interleavings.
                const int max_trial = 100;
                const int min_requirement = 20;
#else
                const int min_requirement = 100;
                const int max_trial = 20;
#endif /* _WIN32||_WIN64 */
                if( PopKind[k]<min_requirement ) {
                    if( trial>=max_trial ) {
                        if( Verbose )
                            REPORT("Warning: %d threads had only %ld pop_if_present operations %s after %d trials (expected at least %d). "
                               "This problem may merely be unlucky scheduling. "
                               "Investigate only if it happens repeatedly.\n",
                               nthread, long(PopKind[k]), k==0?"failed":"succeeded", max_trial, min_requirement);
                        else
                            REPORT("Warning: the number of %s pop_if_present operations is less than expected for %d threads. Investigate if it happens repeatedly.\n",
                               k==0?"failed":"succeeded", nthread );

                    } else {
                        success = false;
                    }
               }
            }
        }
    }
}

class Bar {
    state_t state;
public:
    static size_t construction_num, destruction_num;
    ptrdiff_t my_id;
    Bar() : state(LIVE), my_id(-1) {}
    Bar(size_t _i) : state(LIVE), my_id(_i) { construction_num++; }
    Bar( const Bar& a_bar ) : state(LIVE) {
        ASSERT( a_bar.state==LIVE, NULL );
        my_id = a_bar.my_id;
        construction_num++;
    }
    ~Bar() {
        ASSERT( state==LIVE, NULL );
        state = DEAD;
        my_id = DEAD;
        destruction_num++;
    }
    void operator=( const Bar& a_bar ) {
        ASSERT( a_bar.state==LIVE, NULL );
        ASSERT( state==LIVE, NULL );
        my_id = a_bar.my_id;
    }
    friend bool operator==(const Bar& bar1, const Bar& bar2 ) ;
} ;

size_t Bar::construction_num = 0;
size_t Bar::destruction_num = 0;

bool operator==(const Bar& bar1, const Bar& bar2) {
    ASSERT( bar1.state==LIVE, NULL );
    ASSERT( bar2.state==LIVE, NULL );
    return bar1.my_id == bar2.my_id;
}

class BarIterator
{
    Bar* bar_ptr;
    BarIterator(Bar* bp_) : bar_ptr(bp_) {}
public:
    ~BarIterator() {}
    BarIterator& operator=( const BarIterator& other ) {
        bar_ptr = other.bar_ptr;
        return *this;
    }
    Bar& operator*() const {
        return *bar_ptr;
    }
    BarIterator& operator++() {
        ++bar_ptr;
        return *this;
    }
    Bar* operator++(int) {
        Bar* result = &operator*();
        operator++();
        return result;
    }
    friend bool operator==(const BarIterator& bia, const BarIterator& bib) ;
    friend bool operator!=(const BarIterator& bia, const BarIterator& bib) ;
    template<typename CQ, typename T, typename TIter, typename CQ_EX, typename T_EX>
    friend void TestConstructors ();
} ;

bool operator==(const BarIterator& bia, const BarIterator& bib) {
    return bia.bar_ptr==bib.bar_ptr;
}

bool operator!=(const BarIterator& bia, const BarIterator& bib) {
    return bia.bar_ptr!=bib.bar_ptr;
}

#if TBB_USE_EXCEPTIONS
class Bar_exception : public std::bad_alloc {
public:
    virtual const char *what() const throw() __TBB_override { return "making the entry invalid"; }
    virtual ~Bar_exception() throw() {}
};

class BarEx {
    static int count;
public:
    state_t state;
    typedef enum {
        PREPARATION,
        COPY_CONSTRUCT
    } mode_t;
    static mode_t mode;
    ptrdiff_t my_id;
    ptrdiff_t my_tilda_id;
    static int button;
    BarEx() : state(LIVE), my_id(-1), my_tilda_id(-1) {}
    BarEx(size_t _i) : state(LIVE), my_id(_i), my_tilda_id(my_id^(-1)) {}
    BarEx( const BarEx& a_bar ) : state(LIVE) {
        ASSERT( a_bar.state==LIVE, NULL );
        my_id = a_bar.my_id;
        if( mode==PREPARATION )
            if( !( ++count % 100 ) )
                throw Bar_exception();
        my_tilda_id = a_bar.my_tilda_id;
    }
    ~BarEx() {
        ASSERT( state==LIVE, NULL );
        state = DEAD;
        my_id = DEAD;
    }
    static void set_mode( mode_t m ) { mode = m; }
    void operator=( const BarEx& a_bar ) {
        ASSERT( a_bar.state==LIVE, NULL );
        ASSERT( state==LIVE, NULL );
        my_id = a_bar.my_id;
        my_tilda_id = a_bar.my_tilda_id;
    }
    friend bool operator==(const BarEx& bar1, const BarEx& bar2 ) ;
} ;

int    BarEx::count = 0;
BarEx::mode_t BarEx::mode = BarEx::PREPARATION;

bool operator==(const BarEx& bar1, const BarEx& bar2) {
    ASSERT( bar1.state==LIVE, NULL );
    ASSERT( bar2.state==LIVE, NULL );
    ASSERT( (bar1.my_id ^ bar1.my_tilda_id) == -1, NULL );
    ASSERT( (bar2.my_id ^ bar2.my_tilda_id) == -1, NULL );
    return bar1.my_id==bar2.my_id && bar1.my_tilda_id==bar2.my_tilda_id;
}
#endif /* TBB_USE_EXCEPTIONS */

template<typename CQ, typename T, typename TIter, typename CQ_EX, typename T_EX>
void TestConstructors ()
{
    CQ src_queue;
    typename CQ::const_iterator dqb;
    typename CQ::const_iterator dqe;
    typename CQ::const_iterator iter;

    for( size_t size=0; size<1001; ++size ) {
        for( size_t i=0; i<size; ++i )
            src_queue.push(T(i+(i^size)));
        typename CQ::const_iterator sqb( src_queue.unsafe_begin() );
        typename CQ::const_iterator sqe( src_queue.unsafe_end()   );

        CQ dst_queue(sqb, sqe);

        ASSERT(src_queue.size()==dst_queue.size(), "different size");

        src_queue.clear();
    }

    T bar_array[1001];
    for( size_t size=0; size<1001; ++size ) {
        for( size_t i=0; i<size; ++i )
            bar_array[i] = T(i+(i^size));

        const TIter sab(bar_array+0);
        const TIter sae(bar_array+size);

        CQ dst_queue2(sab, sae);

        ASSERT( size==unsigned(dst_queue2.size()), NULL );
        ASSERT( sab==TIter(bar_array+0), NULL );
        ASSERT( sae==TIter(bar_array+size), NULL );

        dqb = dst_queue2.unsafe_begin();
        dqe = dst_queue2.unsafe_end();
        TIter v_iter(sab);
        for( ; dqb != dqe; ++dqb, ++v_iter )
            ASSERT( *dqb == *v_iter, "unexpected element" );
        ASSERT( v_iter==sae, "different size?" );
    }

    src_queue.clear();

    CQ dst_queue3( src_queue );
    ASSERT( src_queue.size()==dst_queue3.size(), NULL );
    ASSERT( 0==dst_queue3.size(), NULL );

    int k=0;
    for( size_t i=0; i<1001; ++i ) {
        T tmp_bar;
        src_queue.push(T(++k));
        src_queue.push(T(++k));
        src_queue.try_pop(tmp_bar);

        CQ dst_queue4( src_queue );

        ASSERT( src_queue.size()==dst_queue4.size(), NULL );

        dqb = dst_queue4.unsafe_begin();
        dqe = dst_queue4.unsafe_end();
        iter = src_queue.unsafe_begin();

        for( ; dqb != dqe; ++dqb, ++iter )
            ASSERT( *dqb == *iter, "unexpected element" );

        ASSERT( iter==src_queue.unsafe_end(), "different size?" );
    }

    CQ dst_queue5( src_queue );

    ASSERT( src_queue.size()==dst_queue5.size(), NULL );
    dqb = dst_queue5.unsafe_begin();
    dqe = dst_queue5.unsafe_end();
    iter = src_queue.unsafe_begin();
    for( ; dqb != dqe; ++dqb, ++iter )
        ASSERT( *dqb == *iter, "unexpected element" );

    for( size_t i=0; i<100; ++i) {
        T tmp_bar;
        src_queue.push(T(i+1000));
        src_queue.push(T(i+1000));
        src_queue.try_pop(tmp_bar);

        dst_queue5.push(T(i+1000));
        dst_queue5.push(T(i+1000));
        dst_queue5.try_pop(tmp_bar);
    }

    ASSERT( src_queue.size()==dst_queue5.size(), NULL );
    dqb = dst_queue5.unsafe_begin();
    dqe = dst_queue5.unsafe_end();
    iter = src_queue.unsafe_begin();
    for( ; dqb != dqe; ++dqb, ++iter )
        ASSERT( *dqb == *iter, "unexpected element" );
    ASSERT( iter==src_queue.unsafe_end(), "different size?" );

#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN || __TBB_PLACEMENT_NEW_EXCEPTION_SAFETY_BROKEN
    REPORT("Known issue: part of the constructor test is skipped.\n");
#elif TBB_USE_EXCEPTIONS
    k = 0;
    typename CQ_EX::size_type n_elements=0;
    CQ_EX src_queue_ex;
    for( size_t size=0; size<1001; ++size ) {
        T_EX tmp_bar_ex;
        typename CQ_EX::size_type n_successful_pushes=0;
        T_EX::set_mode( T_EX::PREPARATION );
        try {
            src_queue_ex.push(T_EX(k+(k^size)));
            ++n_successful_pushes;
        } catch (...) {
        }
        ++k;
        try {
            src_queue_ex.push(T_EX(k+(k^size)));
            ++n_successful_pushes;
        } catch (...) {
        }
        ++k;
        src_queue_ex.try_pop(tmp_bar_ex);
        n_elements += (n_successful_pushes - 1);
        ASSERT( src_queue_ex.size()==n_elements, NULL);

        T_EX::set_mode( T_EX::COPY_CONSTRUCT );
        CQ_EX dst_queue_ex( src_queue_ex );

        ASSERT( src_queue_ex.size()==dst_queue_ex.size(), NULL );

        typename CQ_EX::const_iterator dqb_ex  = dst_queue_ex.unsafe_begin();
        typename CQ_EX::const_iterator dqe_ex  = dst_queue_ex.unsafe_end();
        typename CQ_EX::const_iterator iter_ex = src_queue_ex.unsafe_begin();

        for( ; dqb_ex != dqe_ex; ++dqb_ex, ++iter_ex )
            ASSERT( *dqb_ex == *iter_ex, "unexpected element" );
        ASSERT( iter_ex==src_queue_ex.unsafe_end(), "different size?" );
    }
#endif /* TBB_USE_EXCEPTIONS */

#if __TBB_CPP11_RVALUE_REF_PRESENT
    // Testing work of move constructors. TODO: merge into TestMoveConstructors?
    src_queue.clear();

    typedef typename CQ::size_type qsize_t;
    for( qsize_t size = 0; size < 1001; ++size ) {
        for( qsize_t i = 0; i < size; ++i )
            src_queue.push( T(i + (i ^ size)) );
        std::vector<const T*> locations(size);
        typename CQ::const_iterator qit = src_queue.unsafe_begin();
        for( qsize_t i = 0; i < size; ++i, ++qit )
            locations[i] = &(*qit);

        qsize_t size_of_queue = src_queue.size();
        CQ dst_queue( std::move(src_queue) );

        ASSERT( src_queue.empty() && src_queue.size() == 0, "not working move constructor?" );
        ASSERT( size == size_of_queue && size_of_queue == dst_queue.size(),
                "not working move constructor?" );

        qit = dst_queue.unsafe_begin();
        for( qsize_t i = 0; i < size; ++i, ++qit )
            ASSERT( locations[i] == &(*qit), "there was data movement during move constructor" );

        for( qsize_t i = 0; i < size; ++i ) {
            T test(i + (i ^ size));
            T popped;
            bool pop_result = dst_queue.try_pop( popped );

            ASSERT( pop_result, NULL );
            ASSERT( test == popped, NULL );
        }
    }
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<class T>
class allocator: public tbb::cache_aligned_allocator<T> {
public:
    size_t m_unique_id;

    allocator() : m_unique_id( 0 ) {}

    allocator(size_t unique_id) { m_unique_id = unique_id; }

    template<typename U>
    allocator(const allocator<U>& a) throw() { m_unique_id = a.m_unique_id; }

    template<typename U>
    struct rebind { typedef allocator<U> other; };

    friend bool operator==(const allocator& lhs, const allocator& rhs) {
        return lhs.m_unique_id == rhs.m_unique_id;
    }
};

// Checks operability of the queue the data was moved from
template<typename T, typename CQ>
void TestQueueOperabilityAfterDataMove( CQ& queue ) {
    const size_t size = 10;
    std::vector<T> v(size);
    for( size_t i = 0; i < size; ++i ) v[i] = T( i * i + i );

    FillTest<push_op>(queue, v);
    EmptyTest<try_pop_op>(queue, v);
    bounded_queue_specific_test(queue, v);
}

template<class CQ, class T>
void TestMoveConstructors() {
    T::construction_num = T::destruction_num = 0;
    CQ src_queue( allocator<T>(0) );
    const size_t size = 10;
    for( size_t i = 0; i < size; ++i )
        src_queue.push( T(i + (i ^ size)) );
    ASSERT( T::construction_num == 2 * size, NULL );
    ASSERT( T::destruction_num == size, NULL );

    const T* locations[size];
    typename CQ::const_iterator qit = src_queue.unsafe_begin();
    for( size_t i = 0; i < size; ++i, ++qit )
        locations[i] = &(*qit);

    // Ensuring allocation operation takes place during move when allocators are different
    T::construction_num = T::destruction_num = 0;
    CQ dst_queue( std::move(src_queue), allocator<T>(1) );
    ASSERT( T::construction_num == size, NULL );
    ASSERT( T::destruction_num == size+1, NULL ); // One item is used by the queue destructor

    TestQueueOperabilityAfterDataMove<T>( src_queue );

    qit = dst_queue.unsafe_begin();
    for( size_t i = 0; i < size; ++i, ++qit ) {
        ASSERT( locations[i] != &(*qit), "an item should have been copied but was not" );
        locations[i] = &(*qit);
    }

    T::construction_num = T::destruction_num = 0;
    // Ensuring there is no allocation operation during move with equal allocators
    CQ dst_queue2( std::move(dst_queue), allocator<T>(1) );
    ASSERT( T::construction_num == 0, NULL );
    ASSERT( T::destruction_num == 0, NULL );

    TestQueueOperabilityAfterDataMove<T>( dst_queue );

    qit = dst_queue2.unsafe_begin();
    for( size_t i = 0; i < size; ++i, ++qit ) {
        ASSERT( locations[i] == &(*qit), "an item should have been moved but was not" );
    }

    for( size_t i = 0; i < size; ++i) {
        T test(i + (i ^ size));
        T popped;
        bool pop_result = dst_queue2.try_pop( popped );
        ASSERT( pop_result, NULL );
        ASSERT( test == popped, NULL );
    }
    ASSERT( dst_queue2.empty(), NULL );
    ASSERT( dst_queue2.size() == 0, NULL );
}

void TestMoveConstruction() {
    REMARK("Testing move constructors with specified allocators...");
    TestMoveConstructors< ConcQWithSizeWrapper< Bar, allocator<Bar> >, Bar >();
    TestMoveConstructors< tbb::concurrent_bounded_queue< Bar, allocator<Bar> >, Bar >();
    // TODO: add tests with movable data
    REMARK(" work\n");
}
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

template<typename Iterator1, typename Iterator2>
void TestIteratorAux( Iterator1 i, Iterator2 j, int size ) {
    Iterator1 old_i; // assigned at first iteration below
    for( int k=0; k<size; ++k ) {
        ASSERT( i!=j, NULL );
        ASSERT( !(i==j), NULL );
        // Test "->"
        ASSERT( k+1==i->serial, NULL );
        if( k&1 ) {
            // Test post-increment
            Foo f = *old_i++;
            ASSERT( k+1==f.serial, NULL );
            // Test assignment
            i = old_i;
        } else {
            // Test pre-increment
            if( k<size-1 ) {
                Foo f = *++i;
                ASSERT( k+2==f.serial, NULL );
            } else ++i;
            // Test assignment
            old_i = i;
        }
    }
    ASSERT( !(i!=j), NULL );
    ASSERT( i==j, NULL );
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
    AssertSameType( static_cast<typename Iterator::iterator_category*>(0), static_cast<std::forward_iterator_tag*>(0) );
    T x;
    typename Iterator::reference xr = x;
    typename Iterator::pointer xp = &x;
    ASSERT( &xr==xp, NULL );
}

//! Test the iterators for concurrent_queue
template<typename CQ>
void TestIterator() {
    CQ queue;
    const CQ& const_queue = queue;
    for( int j=0; j<500; ++j ) {
        TestIteratorAux( queue.unsafe_begin()      , queue.unsafe_end()      , j );
        TestIteratorAux( const_queue.unsafe_begin(), const_queue.unsafe_end(), j );
        TestIteratorAux( const_queue.unsafe_begin(), queue.unsafe_end()      , j );
        TestIteratorAux( queue.unsafe_begin()      , const_queue.unsafe_end(), j );
        Foo f;
        f.serial = j+1;
        queue.push(f);
    }
    TestIteratorAssignment<typename CQ::const_iterator>( const_queue.unsafe_begin() );
    TestIteratorAssignment<typename CQ::const_iterator>( queue.unsafe_begin() );
    TestIteratorAssignment<typename CQ::iterator>( queue.unsafe_begin() );
    TestIteratorTraits<typename CQ::const_iterator, const Foo>();
    TestIteratorTraits<typename CQ::iterator, Foo>();
}

template<typename CQ>
void TestConcurrentQueueType() {
    AssertSameType( typename CQ::value_type(), Foo() );
    Foo f;
    const Foo g;
    typename CQ::reference r = f;
    ASSERT( &r==&f, NULL );
    ASSERT( !r.is_const(), NULL );
    typename CQ::const_reference cr = g;
    ASSERT( &cr==&g, NULL );
    ASSERT( cr.is_const(), NULL );
}

template<typename CQ, typename T>
void TestEmptyQueue() {
    const CQ queue;
    ASSERT( queue.size()==0, NULL );
    ASSERT( queue.capacity()>0, NULL );
    ASSERT( size_t(queue.capacity())>=size_t(-1)/(sizeof(void*)+sizeof(T)), NULL );
}

template<typename CQ,typename T>
void TestFullQueue() {
    for( int n=0; n<10; ++n ) {
        T::clear_counters();
        CQ queue;
        queue.set_capacity(n);
        for( int i=0; i<=n; ++i ) {
            T f;
            f.serial = i;
            bool result = queue.try_push( f );
            ASSERT( result==(i<n), NULL );
        }
        for( int i=0; i<=n; ++i ) {
            T f;
            bool result = queue.try_pop( f );
            ASSERT( result==(i<n), NULL );
            ASSERT( !result || f.serial==i, NULL );
        }
        ASSERT( T::get_n_constructed()==T::get_n_destroyed(), NULL );
    }
}

template<typename CQ>
void TestClear() {
    FooConstructed = 0;
    FooDestroyed = 0;
    const unsigned int n=5;

    CQ queue;
    const int q_capacity=10;
    queue.set_capacity(q_capacity);
    for( size_t i=0; i<n; ++i ) {
        Foo f;
        f.serial = int(i);
        queue.push( f );
    }
    ASSERT( unsigned(queue.size())==n, NULL );
    queue.clear();
    ASSERT( queue.size()==0, NULL );
    for( size_t i=0; i<n; ++i ) {
        Foo f;
        f.serial = int(i);
        queue.push( f );
    }
    ASSERT( unsigned(queue.size())==n, NULL );
    queue.clear();
    ASSERT( queue.size()==0, NULL );
    for( size_t i=0; i<n; ++i ) {
        Foo f;
        f.serial = int(i);
        queue.push( f );
    }
    ASSERT( unsigned(queue.size())==n, NULL );
}

template<typename T>
struct TestNegativeQueueBody: NoAssign {
    tbb::concurrent_bounded_queue<T>& queue;
    const int nthread;
    TestNegativeQueueBody( tbb::concurrent_bounded_queue<T>& q, int n ) : queue(q), nthread(n) {}
    void operator()( int k ) const {
        if( k==0 ) {
            int number_of_pops = nthread-1;
            // Wait for all pops to pend.
            while( queue.size()>-number_of_pops ) {
                __TBB_Yield();
            }
            for( int i=0; ; ++i ) {
                ASSERT( queue.size()==i-number_of_pops, NULL );
                ASSERT( queue.empty()==(queue.size()<=0), NULL );
                if( i==number_of_pops ) break;
                // Satisfy another pop
                queue.push( T() );
            }
        } else {
            // Pop item from queue
            T item;
            queue.pop(item);
        }
    }
};

//! Test a queue with a negative size.
template<typename T>
void TestNegativeQueue( int nthread ) {
    tbb::concurrent_bounded_queue<T> queue;
    NativeParallelFor( nthread, TestNegativeQueueBody<T>(queue,nthread) );
}

#if TBB_USE_EXCEPTIONS
template<template<typename, typename> class CQ,typename A1,typename A2,typename T>
void TestExceptionBody() {
    enum methods {
        m_push = 0,
        m_pop
    };

    REMARK("Testing exception safety\n");
    MaxFooCount = 5;
    // verify 'clear()' on exception; queue's destructor calls its clear()
    // Do test on queues of two different types at the same time to
    // catch problem with incorrect sharing between templates.
    {
        CQ<T,A1> queue0;
        CQ<int,A1> queue1;
        for( int i=0; i<2; ++i ) {
            bool caught = false;
            try {
                // concurrent_queue internally rebinds the allocator to the one for 'char'
                A2::init_counters();
                A2::set_limits(N/2);
                for( int k=0; k<N; k++ ) {
                    if( i==0 )
                        push(queue0, T(), i);
                    else
                        queue1.push( k );
                }
            } catch (...) {
                caught = true;
            }
            ASSERT( caught, "call to push should have thrown exception" );
        }
    }
    REMARK("... queue destruction test passed\n");

    try {
        int n_pushed=0, n_popped=0;
        for(int t = 0; t <= 1; t++)// exception type -- 0 : from allocator(), 1 : from Foo's constructor
        {
            CQ<T,A1> queue_test;
            for( int m=m_push; m<=m_pop; m++ ) {
                // concurrent_queue internally rebinds the allocator to the one for 'char'
                A2::init_counters();

                if(t) MaxFooCount = MaxFooCount + 400;
                else A2::set_limits(N/2);

                try {
                    switch(m) {
                    case m_push:
                        for( int k=0; k<N; k++ ) {
                            push( queue_test, T(), k );
                            n_pushed++;
                        }
                        break;
                    case m_pop:
                        n_popped=0;
                        for( int k=0; k<n_pushed; k++ ) {
                            T elt;
                            queue_test.try_pop( elt );
                            n_popped++;
                        }
                        n_pushed = 0;
                        A2::set_limits();
                        break;
                    }
                    if( !t && m==m_push ) ASSERT(false, "should throw an exception");
                } catch ( Foo_exception & ) {
                    long tc = MaxFooCount;
                    MaxFooCount = 0; // disable exception
                    switch(m) {
                    case m_push:
                        ASSERT( ptrdiff_t(queue_test.size())==n_pushed, "incorrect queue size" );
                        for( int k=0; k<(int)tc; k++ ) {
                            push( queue_test, T(), k );
                            n_pushed++;
                        }
                        break;
                    case m_pop:
                        n_pushed -= (n_popped+1); // including one that threw the exception
                        ASSERT( n_pushed>=0, "n_pushed cannot be less than 0" );
                        for( int k=0; k<1000; k++ ) {
                            push( queue_test, T(), k );
                            n_pushed++;
                        }
                        ASSERT( !queue_test.empty(), "queue must not be empty" );
                        ASSERT( ptrdiff_t(queue_test.size())==n_pushed, "queue size must be equal to n pushed" );
                        for( int k=0; k<n_pushed; k++ ) {
                            T elt;
                            queue_test.try_pop( elt );
                        }
                        ASSERT( queue_test.empty(), "queue must be empty" );
                        ASSERT( queue_test.size()==0, "queue must be empty" );
                        break;
                    }
                    MaxFooCount = tc;
                } catch ( std::bad_alloc & ) {
                    A2::set_limits(); // disable exception from allocator
                    size_t size = queue_test.size();
                    switch(m) {
                    case m_push:
                        ASSERT( size>0, "incorrect queue size");
                        break;
                    case m_pop:
                        if( !t ) ASSERT( false, "should not throw an exception" );
                        break;
                    }
                }
                REMARK("... for t=%d and m=%d, exception test passed\n", t, m);
            }
        }
    } catch(...) {
        ASSERT(false, "unexpected exception");
    }
}
#endif /* TBB_USE_EXCEPTIONS */

void TestExceptions() {
#if __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN
    REPORT("Known issue: exception safety test is skipped.\n");
#elif TBB_USE_EXCEPTIONS
    typedef static_counting_allocator<std::allocator<FooEx>, size_t> allocator_t;
    typedef static_counting_allocator<std::allocator<char>, size_t> allocator_char_t;
    TestExceptionBody<ConcQWithSizeWrapper,allocator_t,allocator_char_t,FooEx>();
    TestExceptionBody<tbb::concurrent_bounded_queue,allocator_t,allocator_char_t,FooEx>();
#endif /* TBB_USE_EXCEPTIONS */
}

template<typename CQ, typename T>
struct TestQueueElements: NoAssign {
    CQ& queue;
    const int nthread;
    TestQueueElements( CQ& q, int n ) : queue(q), nthread(n) {}
    void operator()( int k ) const {
        for( int i=0; i<1000; ++i ) {
            if( (i&0x1)==0 ) {
                ASSERT( T(k)<T(nthread), NULL );
                queue.push( T(k) );
            } else {
                // Pop item from queue
                T item = 0;
                queue.try_pop(item);
                ASSERT( item<=T(nthread), NULL );
            }
        }
    }
};

//! Test concurrent queue with primitive data type
template<typename CQ, typename T>
void TestPrimitiveTypes( int nthread, T exemplar )
{
    CQ queue;
    for( int i=0; i<100; ++i )
        queue.push( exemplar );
    NativeParallelFor( nthread, TestQueueElements<CQ,T>(queue,nthread) );
}

#include "harness_m128.h"

#if HAVE_m128 || HAVE_m256

//! Test concurrent queue with vector types
/** Type Queue should be a queue of ClassWithSSE/ClassWithAVX. */
template<typename ClassWithVectorType, typename Queue>
void TestVectorTypes() {
    Queue q1;
    for( int i=0; i<100; ++i ) {
        // VC8 does not properly align a temporary value; to work around, use explicit variable
        ClassWithVectorType bar(i);
        q1.push(bar);
    }

    // Copy the queue
    Queue q2 = q1;
    // Check that elements of the copy are correct
    typename Queue::const_iterator ci = q2.unsafe_begin();
    for( int i=0; i<100; ++i ) {
        ClassWithVectorType foo = *ci;
        ClassWithVectorType bar(i);
        ASSERT( *ci==bar, NULL );
        ++ci;
    }

    for( int i=0; i<101; ++i ) {
        ClassWithVectorType tmp;
        bool b = q1.try_pop( tmp );
        ASSERT( b==(i<100), NULL );
        ClassWithVectorType bar(i);
        ASSERT( !b || tmp==bar, NULL );
    }
}
#endif /* HAVE_m128 || HAVE_m256 */

void TestEmptiness()
{
    REMARK(" Test Emptiness\n");
    TestEmptyQueue<ConcQWithCapacity<char>, char>();
    TestEmptyQueue<ConcQWithCapacity<Foo>, Foo>();
    TestEmptyQueue<tbb::concurrent_bounded_queue<char>, char>();
    TestEmptyQueue<tbb::concurrent_bounded_queue<Foo>, Foo>();
}

void TestFullness()
{
    REMARK(" Test Fullness\n");
    TestFullQueue<ConcQWithCapacity<Foo>,Foo>();
    TestFullQueue<tbb::concurrent_bounded_queue<Foo>,Foo>();
}

void TestClearWorks()
{
    REMARK(" Test concurrent_queue::clear() works\n");
    TestClear<ConcQWithCapacity<Foo> >();
    TestClear<tbb::concurrent_bounded_queue<Foo> >();
}

void TestQueueTypeDeclaration()
{
    REMARK(" Test concurrent_queue's types work\n");
    TestConcurrentQueueType<tbb::concurrent_queue<Foo> >();
    TestConcurrentQueueType<tbb::concurrent_bounded_queue<Foo> >();
}

void TestQueueIteratorWorks()
{
    REMARK(" Test concurrent_queue's iterators work\n");
    TestIterator<tbb::concurrent_queue<Foo> >();
    TestIterator<tbb::concurrent_bounded_queue<Foo> >();
}

#if TBB_USE_EXCEPTIONS
#define BAR_EX BarEx
#else
#define BAR_EX Empty  /* passed as template arg but should not be used */
#endif
class Empty;

void TestQueueConstructors()
{
    REMARK(" Test concurrent_queue's constructors work\n");
    TestConstructors<ConcQWithSizeWrapper<Bar>,Bar,BarIterator,ConcQWithSizeWrapper<BAR_EX>,BAR_EX>();
    TestConstructors<tbb::concurrent_bounded_queue<Bar>,Bar,BarIterator,tbb::concurrent_bounded_queue<BAR_EX>,BAR_EX>();
}

void TestQueueWorksWithPrimitiveTypes()
{
    REMARK(" Test concurrent_queue works with primitive types\n");
    TestPrimitiveTypes<tbb::concurrent_queue<char>, char>( MaxThread, (char)1 );
    TestPrimitiveTypes<tbb::concurrent_queue<int>, int>( MaxThread, (int)-12 );
    TestPrimitiveTypes<tbb::concurrent_queue<float>, float>( MaxThread, (float)-1.2f );
    TestPrimitiveTypes<tbb::concurrent_queue<double>, double>( MaxThread, (double)-4.3 );
    TestPrimitiveTypes<tbb::concurrent_bounded_queue<char>, char>( MaxThread, (char)1 );
    TestPrimitiveTypes<tbb::concurrent_bounded_queue<int>, int>( MaxThread, (int)-12 );
    TestPrimitiveTypes<tbb::concurrent_bounded_queue<float>, float>( MaxThread, (float)-1.2f );
    TestPrimitiveTypes<tbb::concurrent_bounded_queue<double>, double>( MaxThread, (double)-4.3 );
}

void TestQueueWorksWithSSE()
{
    REMARK(" Test concurrent_queue works with SSE data\n");
#if HAVE_m128
    TestVectorTypes<ClassWithSSE, tbb::concurrent_queue<ClassWithSSE> >();
    TestVectorTypes<ClassWithSSE, tbb::concurrent_bounded_queue<ClassWithSSE> >();
#endif /* HAVE_m128 */
#if HAVE_m256
    if( have_AVX() ) {
        TestVectorTypes<ClassWithAVX, tbb::concurrent_queue<ClassWithAVX> >();
        TestVectorTypes<ClassWithAVX, tbb::concurrent_bounded_queue<ClassWithAVX> >();
    }
#endif /* HAVE_m256 */
}

void TestConcurrentPushPop()
{
    REMARK(" Test concurrent_queue's concurrent push and pop\n");
    for( int nthread=MinThread; nthread<=MaxThread; ++nthread ) {
        REMARK(" Testing with %d thread(s)\n", nthread );
        TestNegativeQueue<Foo>(nthread);
        for( size_t prefill=0; prefill<64; prefill+=(1+prefill/3) ) {
            TestPushPop<ConcQPushPopWrapper<Foo>,Foo>(prefill,ptrdiff_t(-1),nthread);
            TestPushPop<ConcQPushPopWrapper<Foo>,Foo>(prefill,ptrdiff_t(1),nthread);
            TestPushPop<ConcQPushPopWrapper<Foo>,Foo>(prefill,ptrdiff_t(2),nthread);
            TestPushPop<ConcQPushPopWrapper<Foo>,Foo>(prefill,ptrdiff_t(10),nthread);
            TestPushPop<ConcQPushPopWrapper<Foo>,Foo>(prefill,ptrdiff_t(100),nthread);
        }
        for( size_t prefill=0; prefill<64; prefill+=(1+prefill/3) ) {
            TestPushPop<tbb::concurrent_bounded_queue<Foo>,Foo>(prefill,ptrdiff_t(-1),nthread);
            TestPushPop<tbb::concurrent_bounded_queue<Foo>,Foo>(prefill,ptrdiff_t(1),nthread);
            TestPushPop<tbb::concurrent_bounded_queue<Foo>,Foo>(prefill,ptrdiff_t(2),nthread);
            TestPushPop<tbb::concurrent_bounded_queue<Foo>,Foo>(prefill,ptrdiff_t(10),nthread);
            TestPushPop<tbb::concurrent_bounded_queue<Foo>,Foo>(prefill,ptrdiff_t(100),nthread);
        }
    }
}

#if TBB_USE_EXCEPTIONS
tbb::atomic<size_t> num_pushed;
tbb::atomic<size_t> num_popped;
tbb::atomic<size_t> failed_pushes;
tbb::atomic<size_t> failed_pops;

class SimplePushBody {
    tbb::concurrent_bounded_queue<int>* q;
    int max;
public:
    SimplePushBody(tbb::concurrent_bounded_queue<int>* _q, int hi_thr) : q(_q), max(hi_thr) {}
    bool operator()() { // predicate for spin_wait_while
        return q->size()<max;
    }
    void operator()(int thread_id) const {
        if (thread_id == max) {
            spin_wait_while( *this );
            q->abort();
            return;
        }
        try {
            q->push(42);
            ++num_pushed;
        } catch ( tbb::user_abort& ) {
            ++failed_pushes;
        }
    }
};

class SimplePopBody {
    tbb::concurrent_bounded_queue<int>* q;
    int max;
    int prefill;
public:
    SimplePopBody(tbb::concurrent_bounded_queue<int>* _q, int hi_thr, int nitems)
    : q(_q), max(hi_thr), prefill(nitems) {}
    bool operator()() { // predicate for spin_wait_while
        // There should be `max` pops, and `prefill` should succeed
        return q->size()>prefill-max;
    }
    void operator()(int thread_id) const {
        int e;
        if (thread_id == max) {
            spin_wait_while( *this );
            q->abort();
            return;
        }
        try {
            q->pop(e);
            ++num_popped;
        } catch ( tbb::user_abort& ) {
            ++failed_pops;
        }
    }
};
#endif /* TBB_USE_EXCEPTIONS */

void TestAbort() {
#if TBB_USE_EXCEPTIONS
    for (int nthreads=MinThread; nthreads<=MaxThread; ++nthreads) {
        REMARK("Testing Abort on %d thread(s).\n", nthreads);

        REMARK("...testing pushing to zero-sized queue\n");
        tbb::concurrent_bounded_queue<int> iq1;
        iq1.set_capacity(0);
        for (int i=0; i<10; ++i) {
            num_pushed = num_popped = failed_pushes = failed_pops = 0;
            SimplePushBody my_push_body1(&iq1, nthreads);
            NativeParallelFor( nthreads+1, my_push_body1 );
            ASSERT(num_pushed == 0, "no elements should have been pushed to zero-sized queue");
            ASSERT((int)failed_pushes == nthreads, "All threads should have failed to push an element to zero-sized queue");
            // Do not test popping each time in order to test queue destruction with no previous pops
            if (nthreads < (MaxThread+MinThread)/2) {
                int e;
                bool queue_empty = !iq1.try_pop(e);
                ASSERT(queue_empty, "no elements should have been popped from zero-sized queue");
            }
        }

        REMARK("...testing pushing to small-sized queue\n");
        tbb::concurrent_bounded_queue<int> iq2;
        iq2.set_capacity(2);
        for (int i=0; i<10; ++i) {
            num_pushed = num_popped = failed_pushes = failed_pops = 0;
            SimplePushBody my_push_body2(&iq2, nthreads);
            NativeParallelFor( nthreads+1, my_push_body2 );
            ASSERT(num_pushed <= 2, "at most 2 elements should have been pushed to queue of size 2");
            if (nthreads >= 2)
                ASSERT((int)failed_pushes == nthreads-2, "nthreads-2 threads should have failed to push an element to queue of size 2");
            int e;
            while (iq2.try_pop(e)) ;
        }

        REMARK("...testing popping from small-sized queue\n");
        tbb::concurrent_bounded_queue<int> iq3;
        iq3.set_capacity(2);
        for (int i=0; i<10; ++i) {
            num_pushed = num_popped = failed_pushes = failed_pops = 0;
            iq3.push(42);
            iq3.push(42);
            SimplePopBody my_pop_body(&iq3, nthreads, 2);
            NativeParallelFor( nthreads+1, my_pop_body );
            ASSERT(num_popped <= 2, "at most 2 elements should have been popped from queue of size 2");
            if (nthreads >= 2)
                ASSERT((int)failed_pops == nthreads-2, "nthreads-2 threads should have failed to pop an element from queue of size 2");
            else {
                int e;
                iq3.pop(e);
            }
        }

        REMARK("...testing pushing and popping from small-sized queue\n");
        tbb::concurrent_bounded_queue<int> iq4;
        int cap = nthreads/2;
        if (!cap) cap=1;
        iq4.set_capacity(cap);
        for (int i=0; i<10; ++i) {
            num_pushed = num_popped = failed_pushes = failed_pops = 0;
            SimplePushBody my_push_body2(&iq4, nthreads);
            NativeParallelFor( nthreads+1, my_push_body2 );
            ASSERT((int)num_pushed <= cap, "at most cap elements should have been pushed to queue of size cap");
            if (nthreads >= cap)
                ASSERT((int)failed_pushes == nthreads-cap, "nthreads-cap threads should have failed to push an element to queue of size cap");
            SimplePopBody my_pop_body(&iq4, nthreads, (int)num_pushed);
            NativeParallelFor( nthreads+1, my_pop_body );
            ASSERT((int)num_popped <= cap, "at most cap elements should have been popped from queue of size cap");
            if (nthreads >= cap)
                ASSERT((int)failed_pops == nthreads-cap, "nthreads-cap threads should have failed to pop an element from queue of size cap");
            else {
                int e;
                while (iq4.try_pop(e)) ;
            }
        }
    }
#endif
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct MoveOperationTracker {
    static size_t copy_constructor_called_times;
    static size_t move_constructor_called_times;
    static size_t copy_assignment_called_times;
    static size_t move_assignment_called_times;

    MoveOperationTracker() {}
    MoveOperationTracker(const MoveOperationTracker&) {
        ++copy_constructor_called_times;
    }
    MoveOperationTracker(MoveOperationTracker&&) {
        ++move_constructor_called_times;
    }
    MoveOperationTracker& operator=(MoveOperationTracker const&) {
        ++copy_assignment_called_times;
        return *this;
    }
    MoveOperationTracker& operator=(MoveOperationTracker&&) {
        ++move_assignment_called_times;
        return *this;
    }
};
size_t MoveOperationTracker::copy_constructor_called_times = 0;
size_t MoveOperationTracker::move_constructor_called_times = 0;
size_t MoveOperationTracker::copy_assignment_called_times = 0;
size_t MoveOperationTracker::move_assignment_called_times = 0;

template <class CQ, push_t push_op, pop_t pop_op>
void TestMoveSupport() {
    size_t &mcct = MoveOperationTracker::move_constructor_called_times;
    size_t &ccct = MoveOperationTracker::copy_constructor_called_times;
    size_t &cact = MoveOperationTracker::copy_assignment_called_times;
    size_t &mact = MoveOperationTracker::move_assignment_called_times;
    mcct = ccct = cact = mact = 0;

    CQ q;

    ASSERT(mcct == 0, "Value must be zero-initialized");
    ASSERT(ccct == 0, "Value must be zero-initialized");
    ASSERT(pusher<push_op>::push( q, MoveOperationTracker() ), NULL);
    ASSERT(mcct == 1, "Not working push(T&&) or try_push(T&&)?");
    ASSERT(ccct == 0, "Copying of arg occurred during push(T&&) or try_push(T&&)");

    MoveOperationTracker ob;
    ASSERT(pusher<push_op>::push( q, std::move(ob) ), NULL);
    ASSERT(mcct == 2, "Not working push(T&&) or try_push(T&&)?");
    ASSERT(ccct == 0, "Copying of arg occurred during push(T&&) or try_push(T&&)");

    ASSERT(cact == 0, "Copy assignment called during push(T&&) or try_push(T&&)");
    ASSERT(mact == 0, "Move assignment called during push(T&&) or try_push(T&&)");

    bool result = popper<pop_op>::pop( q, ob );
    ASSERT(result, NULL);
    ASSERT(cact == 0, "Copy assignment called during try_pop(T&&)");
    ASSERT(mact == 1, "Move assignment was not called during try_pop(T&&)");
}

void TestMoveSupportInPushPop() {
    REMARK("Testing Move Support in Push/Pop...");
    TestMoveSupport< tbb::concurrent_queue<MoveOperationTracker>, push_op, try_pop_op >();
    TestMoveSupport< tbb::concurrent_bounded_queue<MoveOperationTracker>, push_op, pop_op >();
    TestMoveSupport< tbb::concurrent_bounded_queue<MoveOperationTracker>, try_push_op, try_pop_op >();
    REMARK(" works.\n");
}

#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
class NonTrivialConstructorType {
public:
    NonTrivialConstructorType( int a = 0 ) : m_a( a ), m_str( "" ) {}
    NonTrivialConstructorType( const std::string& str ) : m_a( 0 ), m_str( str ) {}
    NonTrivialConstructorType( int a, const std::string& str ) : m_a( a ), m_str( str ) {}
    int get_a() const { return m_a; }
    std::string get_str() const { return m_str; }
private:
    int m_a;
    std::string m_str;
};

enum emplace_t { emplace_op, try_emplace_op };

template< emplace_t emplace_op >
struct emplacer {
    template< typename CQ, typename... Args>
    static void emplace( CQ& queue, Args&&... val ) { queue.emplace( std::forward<Args>( val )... ); }
};

template<>
struct emplacer< try_emplace_op > {
    template<typename CQ, typename... Args>
    static void emplace( CQ& queue, Args&&... val ) {
        bool result = queue.try_emplace( std::forward<Args>( val )... );
        ASSERT( result, "try_emplace error\n" );
    }
};

template<typename CQ, emplace_t emplace_op>
void TestEmplaceInQueue() {
    CQ cq;
    std::string test_str = "I'm being emplaced!";
    {
        emplacer<emplace_op>::emplace( cq, 5 );
        ASSERT( cq.size() == 1, NULL );
        NonTrivialConstructorType popped( -1 );
        bool result = cq.try_pop( popped );
        ASSERT( result, NULL );
        ASSERT( popped.get_a() == 5, NULL );
        ASSERT( popped.get_str() == std::string( "" ), NULL );
    }

    ASSERT( cq.empty(), NULL );

    {
        NonTrivialConstructorType popped( -1 );
        emplacer<emplace_op>::emplace( cq, std::string(test_str) );
        bool result = cq.try_pop( popped );
        ASSERT( result, NULL );
        ASSERT( popped.get_a() == 0, NULL );
        ASSERT( popped.get_str() == test_str, NULL );
    }

    ASSERT( cq.empty(), NULL );

    {
        NonTrivialConstructorType popped( -1, "" );
        emplacer<emplace_op>::emplace( cq, 5, std::string(test_str) );
        bool result = cq.try_pop( popped );
        ASSERT( result, NULL );
        ASSERT( popped.get_a() == 5, NULL );
        ASSERT( popped.get_str() == test_str, NULL );
    }
}
void TestEmplace() {
    REMARK("Testing support for 'emplace' method...");
    TestEmplaceInQueue< ConcQWithSizeWrapper<NonTrivialConstructorType>, emplace_op >();
    TestEmplaceInQueue< tbb::concurrent_bounded_queue<NonTrivialConstructorType>, emplace_op >();
    TestEmplaceInQueue< tbb::concurrent_bounded_queue<NonTrivialConstructorType>, try_emplace_op >();
    REMARK(" works.\n");
}
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

template <typename Queue>
void Examine(Queue q, const std::vector<typename Queue::value_type> &vec) {
    typedef typename Queue::value_type value_type;

    AssertEquality(q, vec);

    const Queue cq = q;
    AssertEquality(cq, vec);

    q.clear();
    AssertEmptiness(q);

    FillTest<push_op>(q, vec);
    EmptyTest<try_pop_op>(q, vec);

    bounded_queue_specific_test(q, vec);

    typename Queue::allocator_type a = q.get_allocator();
    value_type *ptr = a.allocate(1);
    ASSERT(ptr, NULL);
    a.deallocate(ptr, 1);
}

template <typename Queue, typename QueueDebugAlloc>
void TypeTester(const std::vector<typename Queue::value_type> &vec) {
    typedef typename std::vector<typename Queue::value_type>::const_iterator iterator;
    ASSERT(vec.size() >= 5, "Array should have at least 5 elements");
    // Construct an empty queue.
    Queue q1;
    for (iterator it = vec.begin(); it != vec.end(); ++it) q1.push(*it);
    Examine(q1, vec);
    // Copying constructor.
    Queue q3(q1);
    Examine(q3, vec);
    // Construct with non-default allocator.
    QueueDebugAlloc q4;
    for (iterator it = vec.begin(); it != vec.end(); ++it) q4.push(*it);
    Examine(q4, vec);
    // Copying constructor with the same allocator type.
    QueueDebugAlloc q5(q4);
    Examine(q5, vec);
    // Construction with given allocator instance.
    typename QueueDebugAlloc::allocator_type a;
    QueueDebugAlloc q6(a);
    for (iterator it = vec.begin(); it != vec.end(); ++it) q6.push(*it);
    Examine(q6, vec);
    // Construction with copying iteration range and given allocator instance.
    QueueDebugAlloc q7(q1.unsafe_begin(), q1.unsafe_end(), a);
    Examine<QueueDebugAlloc>(q7, vec);
}

template <typename value_type>
void TestTypes(const std::vector<value_type> &vec) {
    TypeTester< ConcQWithSizeWrapper<value_type>, ConcQWithSizeWrapper<value_type, debug_allocator<value_type> > >(vec);
    TypeTester< tbb::concurrent_bounded_queue<value_type>, tbb::concurrent_bounded_queue<value_type, debug_allocator<value_type> > >(vec);
}

void TestTypes() {
    const int NUMBER = 10;

    std::vector<int> arrInt;
    for (int i = 0; i < NUMBER; ++i) arrInt.push_back(i);
    std::vector< tbb::atomic<int> > arrTbb;
    for (int i = 0; i < NUMBER; ++i) {
        tbb::atomic<int> a;
        a = i;
        arrTbb.push_back(a);
    }
    TestTypes(arrInt);
    TestTypes(arrTbb);

#if __TBB_CPP11_SMART_POINTERS_PRESENT
    std::vector< std::shared_ptr<int> > arrShr;
    for (int i = 0; i < NUMBER; ++i) arrShr.push_back(std::make_shared<int>(i));
    std::vector< std::weak_ptr<int> > arrWk;
    std::copy(arrShr.begin(), arrShr.end(), std::back_inserter(arrWk));
    TestTypes(arrShr);
    TestTypes(arrWk);
#else
    REPORT("Known issue: C++11 smart pointer tests are skipped.\n");
#endif /* __TBB_CPP11_SMART_POINTERS_PRESENT */
}

int TestMain () {
    TestEmptiness();

    TestFullness();

    TestClearWorks();

    TestQueueTypeDeclaration();

    TestQueueIteratorWorks();

    TestQueueConstructors();

    TestQueueWorksWithPrimitiveTypes();

    TestQueueWorksWithSSE();

    // Test concurrent operations
    TestConcurrentPushPop();

    TestExceptions();

    TestAbort();

#if __TBB_CPP11_RVALUE_REF_PRESENT
    TestMoveSupportInPushPop();
    TestMoveConstruction();
#if __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT
    TestEmplace();
#endif /* __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
#endif /* __TBB_CPP11_RVALUE_REF_PRESENT */

    TestTypes();

    return Harness::Done;
}
