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

#if (_MSC_VER)
    //MSVC 10 "deprecated" application of some std:: algorithms to raw pointers as not safe.
    //The reason is that destination is not checked against bounds/having enough place.
    #define _SCL_SECURE_NO_WARNINGS
#endif

#include "tbb/concurrent_vector.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_exception.h"
#include "tbb_misc.h"
#include "itt_notify.h"

#include <cstring>
#include <memory> //for uninitialized_fill_n

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

using namespace std;

namespace tbb {

namespace internal {
class concurrent_vector_base_v3::helper :no_assign {
public:
    //! memory page size
    static const size_type page_size = 4096;

    inline static bool incompact_predicate(size_type size) { // assert size != 0, see source/test/test_vector_layout.cpp
        return size < page_size || ((size-1)%page_size < page_size/2 && size < page_size * 128); // for more details
    }

    inline static size_type find_segment_end(const concurrent_vector_base_v3 &v) {
        segment_t *s = v.my_segment;
        segment_index_t u = s==v.my_storage? pointers_per_short_table : pointers_per_long_table;
        segment_index_t k = 0;
        while( k < u && (s[k].load<relaxed>()==segment_allocated() ))
            ++k;
        return k;
    }

    // TODO: optimize accesses to my_first_block
    //! assign first segment size. k - is index of last segment to be allocated, not a count of segments
    inline static void assign_first_segment_if_necessary(concurrent_vector_base_v3 &v, segment_index_t k) {
        if( !v.my_first_block ) {
            /* There was a suggestion to set first segment according to incompact_predicate:
            while( k && !helper::incompact_predicate(segment_size( k ) * element_size) )
                --k; // while previous vector size is compact, decrement
            // reasons to not do it:
            // * constructor(n) is not ready to accept fragmented segments
            // * backward compatibility due to that constructor
            // * current version gives additional guarantee and faster init.
            // * two calls to reserve() will give the same effect.
            */
            v.my_first_block.compare_and_swap(k+1, 0); // store number of segments
        }
    }

    inline static void *allocate_segment(concurrent_vector_base_v3 &v, size_type n) {
        void *ptr = v.vector_allocator_ptr(v, n);
        if(!ptr) throw_exception(eid_bad_alloc); // check for bad allocation, throw exception
        return ptr;
    }

    //! Publish segment so other threads can see it.
    template<typename argument_type>
    inline static void publish_segment( segment_t& s, argument_type rhs ) {
        // see also itt_store_pointer_with_release_v3()
        ITT_NOTIFY( sync_releasing, &s );
        s.store<release>(rhs);
    }

    static size_type enable_segment(concurrent_vector_base_v3 &v, size_type k, size_type element_size, bool mark_as_not_used_on_failure = false);

    // TODO: rename as get_segments_table() and return segment pointer
    inline static void extend_table_if_necessary(concurrent_vector_base_v3 &v, size_type k, size_type start ) {
        if(k >= pointers_per_short_table && v.my_segment == v.my_storage)
            extend_segment_table(v, start );
    }

    static void extend_segment_table(concurrent_vector_base_v3 &v, size_type start);

    struct segment_not_used_predicate: no_assign {
        segment_t &s;
        segment_not_used_predicate(segment_t &segment) : s(segment) {}
        bool operator()() const { return s.load<relaxed>() == segment_not_used ();}
    };
    inline static segment_t& acquire_segment(concurrent_vector_base_v3 &v, size_type index, size_type element_size, bool owner) {
        segment_t &s = v.my_segment[index]; // TODO: pass v.my_segment as argument
        if( s.load<acquire>() == segment_not_used() ) { // do not check for segment_allocation_failed state
            if( owner ) {
                enable_segment( v, index, element_size );
            } else {
                ITT_NOTIFY(sync_prepare, &s);
                spin_wait_while(segment_not_used_predicate(s));
                ITT_NOTIFY(sync_acquired, &s);
            }
        } else {
            ITT_NOTIFY(sync_acquired, &s);
        }
        enforce_segment_allocated(s.load<relaxed>()); //it's hard to recover correctly after segment_allocation_failed state
        return s;
    }

    ///// non-static fields of helper for exception-safe iteration across segments
    segment_t *table;// TODO: review all segment_index_t as just short type
    size_type first_block, k, sz, start, finish, element_size;
    helper(segment_t *segments, size_type fb, size_type esize, size_type index, size_type s, size_type f) throw()
        : table(segments), first_block(fb), k(index), sz(0), start(s), finish(f), element_size(esize) {}
    inline void first_segment() throw() {
        __TBB_ASSERT( start <= finish, NULL );
        __TBB_ASSERT( first_block || !finish, NULL );
        if( k < first_block ) k = 0; // process solid segment at a time
        size_type base = segment_base( k );
        __TBB_ASSERT( base <= start, NULL );
        finish -= base; start -= base; // rebase as offsets from segment k
        sz = k ? base : segment_size( first_block ); // sz==base for k>0
    }
    inline void next_segment() throw() {
        finish -= sz; start = 0; // offsets from next segment
        if( !k ) k = first_block;
        else { ++k; sz = segment_size( k ); }
    }
    template<typename F>
    inline size_type apply(const F &func) {
        first_segment();
        while( sz < finish ) { // work for more than one segment
            //TODO: remove extra load() of table[k] inside func
            func( table[k], table[k].load<relaxed>().pointer<char>() + element_size*start, sz - start );
            next_segment();
        }
        func( table[k], table[k].load<relaxed>().pointer<char>() + element_size*start, finish - start );
        return k;
    }
    inline segment_value_t get_segment_value(size_type index, bool wait) {
        segment_t &s = table[index];
        if( wait && (s.load<acquire>() == segment_not_used()) ) {
            ITT_NOTIFY(sync_prepare, &s);
            spin_wait_while(segment_not_used_predicate(s));
            ITT_NOTIFY(sync_acquired, &s);
        }
        return s.load<relaxed>();
    }
    ~helper() {
        if( sz >= finish ) return; // the work is done correctly
        cleanup();
    }

    //! Out of line code to assists destructor in infrequent cases.
    void cleanup();

    /// TODO: turn into lambda functions when available
    struct init_body {
        internal_array_op2 func;
        const void *arg;
        init_body(internal_array_op2 init, const void *src) : func(init), arg(src) {}
        void operator()(segment_t &, void *begin, size_type n) const {
            func( begin, arg, n );
        }
    };
    struct safe_init_body {
        internal_array_op2 func;
        const void *arg;
        safe_init_body(internal_array_op2 init, const void *src) : func(init), arg(src) {}
        void operator()(segment_t &s, void *begin, size_type n) const {
            enforce_segment_allocated(s.load<relaxed>());
            func( begin, arg, n );
        }
    };
    struct destroy_body {
        internal_array_op1 func;
        destroy_body(internal_array_op1 destroy) : func(destroy) {}
        void operator()(segment_t &s, void *begin, size_type n) const {
            if(s.load<relaxed>() == segment_allocated())
                func( begin, n );
        }
    };
}; // class helper

void concurrent_vector_base_v3::helper::extend_segment_table(concurrent_vector_base_v3 &v, concurrent_vector_base_v3::size_type start) {
    if( start > segment_size(pointers_per_short_table) ) start = segment_size(pointers_per_short_table);
    // If other threads are trying to set pointers in the short segment, wait for them to finish their
    // assignments before we copy the short segment to the long segment. Note: grow_to_at_least depends on it
    for( segment_index_t i = 0; segment_base(i) < start && v.my_segment == v.my_storage; i++ ){
        if(v.my_storage[i].load<relaxed>() == segment_not_used()) {
            ITT_NOTIFY(sync_prepare, &v.my_storage[i]);
            atomic_backoff backoff(true);
            while( v.my_segment == v.my_storage && (v.my_storage[i].load<relaxed>() == segment_not_used()) )
                backoff.pause();
            ITT_NOTIFY(sync_acquired, &v.my_storage[i]);
        }
    }
    if( v.my_segment != v.my_storage ) return;

    segment_t* new_segment_table = (segment_t*)NFS_Allocate( pointers_per_long_table, sizeof(segment_t), NULL );
    __TBB_ASSERT(new_segment_table, "NFS_Allocate should throws exception if it cannot allocate the requested storage, and not returns zero pointer" );
    std::uninitialized_fill_n(new_segment_table,size_t(pointers_per_long_table),segment_t()); //init newly allocated table
   //TODO: replace with static assert
    __TBB_STATIC_ASSERT(pointers_per_long_table >= pointers_per_short_table, "size of the big table should be not lesser than of the small one, as we copy values to it" );
    std::copy(v.my_storage, v.my_storage+pointers_per_short_table, new_segment_table);//copy values from old table, here operator= of segment_t is used
    if( v.my_segment.compare_and_swap( new_segment_table, v.my_storage ) != v.my_storage )
        NFS_Free( new_segment_table );
    // else TODO: add ITT_NOTIFY signals for v.my_segment?
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::helper::enable_segment(concurrent_vector_base_v3 &v, concurrent_vector_base_v3::size_type k, concurrent_vector_base_v3::size_type element_size,
        bool mark_as_not_used_on_failure ) {

    struct segment_scope_guard : no_copy{
        segment_t* my_segment_ptr;
        bool my_mark_as_not_used;
        segment_scope_guard(segment_t& segment, bool mark_as_not_used) : my_segment_ptr(&segment), my_mark_as_not_used(mark_as_not_used){}
        void dismiss(){ my_segment_ptr = 0;}
        ~segment_scope_guard(){
            if (my_segment_ptr){
                if (!my_mark_as_not_used){
                    publish_segment(*my_segment_ptr, segment_allocation_failed());
                }else{
                    publish_segment(*my_segment_ptr, segment_not_used());
                }
            }
        }
    };

    segment_t* s = v.my_segment; // TODO: optimize out as argument? Optimize accesses to my_first_block
    __TBB_ASSERT(s[k].load<relaxed>() != segment_allocated(), "concurrent operation during growth?");

    size_type size_of_enabled_segment =  segment_size(k);
    size_type size_to_allocate = size_of_enabled_segment;
    if( !k ) {
        assign_first_segment_if_necessary(v, default_initial_segments-1);
        size_of_enabled_segment =  2 ;
        size_to_allocate = segment_size(v.my_first_block);

    } else  {
        spin_wait_while_eq( v.my_first_block, segment_index_t(0) );
    }

    if( k && (k < v.my_first_block)){ //no need to allocate anything
        // s[0].array is changed only once ( 0 -> !0 ) and points to uninitialized memory
        segment_value_t array0 = s[0].load<acquire>();
        if(array0 == segment_not_used()){
            // sync_prepare called only if there is a wait
            ITT_NOTIFY(sync_prepare, &s[0]);
            spin_wait_while( segment_not_used_predicate(s[0]));
            array0 = s[0].load<acquire>();
        }
        ITT_NOTIFY(sync_acquired, &s[0]);

        segment_scope_guard k_segment_guard(s[k], false);
        enforce_segment_allocated(array0); // initial segment should be allocated
        k_segment_guard.dismiss();

        publish_segment( s[k],
            static_cast<void*>(array0.pointer<char>() + segment_base(k)*element_size )
        );
    } else {
        segment_scope_guard k_segment_guard(s[k], mark_as_not_used_on_failure);
        publish_segment(s[k], allocate_segment(v, size_to_allocate));
        k_segment_guard.dismiss();
    }
    return size_of_enabled_segment;
}

void concurrent_vector_base_v3::helper::cleanup() {
    if( !sz ) { // allocation failed, restore the table
        segment_index_t k_start = k, k_end = segment_index_of(finish-1);
        if( segment_base( k_start ) < start )
            get_segment_value(k_start++, true); // wait
        if( k_start < first_block ) {
            segment_value_t segment0 = get_segment_value(0, start>0); // wait if necessary
            if((segment0 != segment_not_used()) && !k_start ) ++k_start;
            if(segment0 != segment_allocated())
                for(; k_start < first_block && k_start <= k_end; ++k_start )
                    publish_segment(table[k_start], segment_allocation_failed());
            else for(; k_start < first_block && k_start <= k_end; ++k_start )
                    publish_segment(table[k_start], static_cast<void*>(
                        (segment0.pointer<char>()) + segment_base(k_start)*element_size) );
        }
        for(; k_start <= k_end; ++k_start ) // not in first block
            if(table[k_start].load<acquire>() == segment_not_used())
                publish_segment(table[k_start], segment_allocation_failed());
        // fill allocated items
        first_segment();
        goto recover;
    }
    while( sz <= finish ) { // there is still work for at least one segment
        next_segment();
recover:
        segment_value_t array = table[k].load<relaxed>();
        if(array == segment_allocated())
            std::memset( (array.pointer<char>()) + element_size*start, 0, ((sz<finish?sz:finish) - start)*element_size );
        else __TBB_ASSERT( array == segment_allocation_failed(), NULL );
    }
}

concurrent_vector_base_v3::~concurrent_vector_base_v3() {
    segment_t* s = my_segment;
    if( s != my_storage ) {
#if TBB_USE_ASSERT
        //to please assert in segment_t destructor
        std::fill_n(my_storage,size_t(pointers_per_short_table),segment_t());
#endif /* TBB_USE_ASSERT */
#if TBB_USE_DEBUG
        for( segment_index_t i = 0; i < pointers_per_long_table; i++)
            __TBB_ASSERT( my_segment[i].load<relaxed>() != segment_allocated(), "Segment should have been freed. Please recompile with new TBB before using exceptions.");
#endif
        my_segment = my_storage;
        NFS_Free( s );
    }
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_capacity() const {
    return segment_base( helper::find_segment_end(*this) );
}

void concurrent_vector_base_v3::internal_throw_exception(size_type t) const {
    switch(t) {
        case 0: throw_exception(eid_out_of_range);
        case 1: throw_exception(eid_segment_range_error);
        case 2: throw_exception(eid_index_range_error);
    }
}

void concurrent_vector_base_v3::internal_reserve( size_type n, size_type element_size, size_type max_size ) {
    if( n>max_size )
        throw_exception(eid_reservation_length_error);
    __TBB_ASSERT( n, NULL );
    helper::assign_first_segment_if_necessary(*this, segment_index_of(n-1));
    segment_index_t k = helper::find_segment_end(*this);

    for( ; segment_base(k)<n; ++k ) {
        helper::extend_table_if_necessary(*this, k, 0);
        if(my_segment[k].load<relaxed>() != segment_allocated())
            helper::enable_segment(*this, k, element_size, true ); //in case of failure mark segments as not used
    }
}

//TODO: Looks like atomic loads can be done relaxed here, as the only place this method is called from
//is the constructor, which does not require synchronization (for more details see comment in the
// concurrent_vector_base constructor).
void concurrent_vector_base_v3::internal_copy( const concurrent_vector_base_v3& src, size_type element_size, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    __TBB_ASSERT( my_segment == my_storage, NULL);
    if( n ) {
        helper::assign_first_segment_if_necessary(*this, segment_index_of(n-1));
        size_type b;
        for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
            if( (src.my_segment.load<acquire>() == src.my_storage && k >= pointers_per_short_table)
                || (src.my_segment[k].load<relaxed>() != segment_allocated())) {
                my_early_size = b; break;
            }
            helper::extend_table_if_necessary(*this, k, 0);
            size_type m = helper::enable_segment(*this, k, element_size);
            if( m > n-b ) m = n-b;
            my_early_size = b+m;
            copy( my_segment[k].load<relaxed>().pointer<void>(), src.my_segment[k].load<relaxed>().pointer<void>(), m );
        }
    }
}

void concurrent_vector_base_v3::internal_assign( const concurrent_vector_base_v3& src, size_type element_size, internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    while( my_early_size>n ) { // TODO: improve
        segment_index_t k = segment_index_of( my_early_size-1 );
        size_type b=segment_base(k);
        size_type new_end = b>=n ? b : n;
        __TBB_ASSERT( my_early_size>new_end, NULL );
        enforce_segment_allocated(my_segment[k].load<relaxed>()); //if vector was broken before
        // destructors are supposed to not throw any exceptions
        destroy( my_segment[k].load<relaxed>().pointer<char>() + element_size*(new_end-b), my_early_size-new_end );
        my_early_size = new_end;
    }
    size_type dst_initialized_size = my_early_size;
    my_early_size = n;
    helper::assign_first_segment_if_necessary(*this, segment_index_of(n));
    size_type b;
    for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
        if( (src.my_segment.load<acquire>() == src.my_storage && k >= pointers_per_short_table)
            || src.my_segment[k].load<relaxed>() != segment_allocated() ) { // if source is damaged
                my_early_size = b; break; // TODO: it may cause undestructed items
        }
        helper::extend_table_if_necessary(*this, k, 0);
        if( my_segment[k].load<relaxed>() == segment_not_used())
            helper::enable_segment(*this, k, element_size);
        else
            enforce_segment_allocated(my_segment[k].load<relaxed>());
        size_type m = k? segment_size(k) : 2;
        if( m > n-b ) m = n-b;
        size_type a = 0;
        if( dst_initialized_size>b ) {
            a = dst_initialized_size-b;
            if( a>m ) a = m;
            assign( my_segment[k].load<relaxed>().pointer<void>(), src.my_segment[k].load<relaxed>().pointer<void>(), a );
            m -= a;
            a *= element_size;
        }
        if( m>0 )
            copy( my_segment[k].load<relaxed>().pointer<char>() + a, src.my_segment[k].load<relaxed>().pointer<char>() + a, m );
    }
    __TBB_ASSERT( src.my_early_size==n, "detected use of concurrent_vector::operator= with right side that was concurrently modified" );
}

void* concurrent_vector_base_v3::internal_push_back( size_type element_size, size_type& index ) {
    __TBB_ASSERT( sizeof(my_early_size)==sizeof(uintptr_t), NULL );
    size_type tmp = my_early_size.fetch_and_increment<acquire>();
    index = tmp;
    segment_index_t k_old = segment_index_of( tmp );
    size_type base = segment_base(k_old);
    helper::extend_table_if_necessary(*this, k_old, tmp);
    segment_t& s = helper::acquire_segment(*this, k_old, element_size, base==tmp);
    size_type j_begin = tmp-base;
    return (void*)(s.load<relaxed>().pointer<char>() + element_size*j_begin);
}

void concurrent_vector_base_v3::internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op2 init, const void *src ) {
    internal_grow_to_at_least_with_result( new_size, element_size, init, src );
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_grow_to_at_least_with_result( size_type new_size, size_type element_size, internal_array_op2 init, const void *src ) {
    size_type e = my_early_size;
    while( e<new_size ) {
        size_type f = my_early_size.compare_and_swap(new_size,e);
        if( f==e ) {
            internal_grow( e, new_size, element_size, init, src );
            break;
        }
        e = f;
    }
    // Check/wait for segments allocation completes
    segment_index_t i, k_old = segment_index_of( new_size-1 );
    if( k_old >= pointers_per_short_table && my_segment == my_storage ) {
        spin_wait_while_eq( my_segment, my_storage );
    }
    for( i = 0; i <= k_old; ++i ) {
        segment_t &s = my_segment[i];
        if(s.load<relaxed>() == segment_not_used()) {
            ITT_NOTIFY(sync_prepare, &s);
            atomic_backoff backoff(true);
            while( my_segment[i].load<acquire>() == segment_not_used() ) // my_segment may change concurrently
                backoff.pause();
            ITT_NOTIFY(sync_acquired, &s);
        }
        enforce_segment_allocated(my_segment[i].load<relaxed>());
    }
#if TBB_USE_DEBUG
    size_type capacity = internal_capacity();
    __TBB_ASSERT( capacity >= new_size, NULL);
#endif
    return e;
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_grow_by( size_type delta, size_type element_size, internal_array_op2 init, const void *src ) {
    size_type result = my_early_size.fetch_and_add(delta);
    internal_grow( result, result+delta, element_size, init, src );
    return result;
}

void concurrent_vector_base_v3::internal_grow( const size_type start, size_type finish, size_type element_size, internal_array_op2 init, const void *src ) {
    __TBB_ASSERT( start<finish, "start must be less than finish" );
    segment_index_t k_start = segment_index_of(start), k_end = segment_index_of(finish-1);
    helper::assign_first_segment_if_necessary(*this, k_end);
    helper::extend_table_if_necessary(*this, k_end, start);
    helper range(my_segment, my_first_block, element_size, k_start, start, finish);
    for(; k_end > k_start && k_end >= range.first_block; --k_end ) // allocate segments in reverse order
        helper::acquire_segment(*this, k_end, element_size, true/*for k_end>k_start*/);
    for(; k_start <= k_end; ++k_start ) // but allocate first block in straight order
        helper::acquire_segment(*this, k_start, element_size, segment_base( k_start ) >= start );
    range.apply( helper::init_body(init, src) );
}

void concurrent_vector_base_v3::internal_resize( size_type n, size_type element_size, size_type max_size, const void *src,
                                                internal_array_op1 destroy, internal_array_op2 init ) {
    size_type j = my_early_size;
    if( n > j ) { // construct items
        internal_reserve(n, element_size, max_size);
        my_early_size = n;
        helper for_each(my_segment, my_first_block, element_size, segment_index_of(j), j, n);
        for_each.apply( helper::safe_init_body(init, src) );
    } else {
        my_early_size = n;
        helper for_each(my_segment, my_first_block, element_size, segment_index_of(n), n, j);
        for_each.apply( helper::destroy_body(destroy) );
    }
}

concurrent_vector_base_v3::segment_index_t concurrent_vector_base_v3::internal_clear( internal_array_op1 destroy ) {
    __TBB_ASSERT( my_segment, NULL );
    size_type j = my_early_size;
    my_early_size = 0;
    helper for_each(my_segment, my_first_block, 0, 0, 0, j); // element_size is safe to be zero if 'start' is zero
    j = for_each.apply( helper::destroy_body(destroy) );
    size_type i = helper::find_segment_end(*this);
    return j < i? i : j+1;
}

void *concurrent_vector_base_v3::internal_compact( size_type element_size, void *table, internal_array_op1 destroy, internal_array_op2 copy )
{
    const size_type my_size = my_early_size;
    const segment_index_t k_end = helper::find_segment_end(*this); // allocated segments
    const segment_index_t k_stop = my_size? segment_index_of(my_size-1) + 1 : 0; // number of segments to store existing items: 0=>0; 1,2=>1; 3,4=>2; [5-8]=>3;..
    const segment_index_t first_block = my_first_block; // number of merged segments, getting values from atomics

    segment_index_t k = first_block;
    if(k_stop < first_block)
        k = k_stop;
    else
        while (k < k_stop && helper::incompact_predicate(segment_size( k ) * element_size) ) k++;
    if(k_stop == k_end && k == first_block)
        return NULL;

    segment_t *const segment_table = my_segment;
    internal_segments_table &old = *static_cast<internal_segments_table*>( table );
    //this call is left here for sake of backward compatibility, and as a placeholder for table initialization
    std::fill_n(old.table,sizeof(old.table)/sizeof(old.table[0]),segment_t());
    old.first_block=0;

    if ( k != first_block && k ) // first segment optimization
    {
        // exception can occur here
        void *seg = helper::allocate_segment(*this, segment_size(k));
        old.table[0].store<relaxed>(seg);
        old.first_block = k; // fill info for freeing new segment if exception occurs
        // copy items to the new segment
        size_type my_segment_size = segment_size( first_block );
        for (segment_index_t i = 0, j = 0; i < k && j < my_size; j = my_segment_size) {
            __TBB_ASSERT( segment_table[i].load<relaxed>() == segment_allocated(), NULL);
            void *s = static_cast<void*>(
                static_cast<char*>(seg) + segment_base(i)*element_size );
            //TODO: refactor to use std::min
            if(j + my_segment_size >= my_size) my_segment_size = my_size - j;
            __TBB_TRY { // exception can occur here
                copy( s, segment_table[i].load<relaxed>().pointer<void>(), my_segment_size );
            } __TBB_CATCH(...) { // destroy all the already copied items
                helper for_each(&old.table[0], old.first_block, element_size,
                    0, 0, segment_base(i)+ my_segment_size);
                for_each.apply( helper::destroy_body(destroy) );
                __TBB_RETHROW();
            }
            my_segment_size = i? segment_size( ++i ) : segment_size( i = first_block );
        }
        // commit the changes
        std::copy(segment_table,segment_table + k,old.table);
        for (segment_index_t i = 0; i < k; i++) {
            segment_table[i].store<relaxed>(static_cast<void*>(
                static_cast<char*>(seg) + segment_base(i)*element_size ));
        }
        old.first_block = first_block; my_first_block = k; // now, first_block != my_first_block
        // destroy original copies
        my_segment_size = segment_size( first_block ); // old.first_block actually
        for (segment_index_t i = 0, j = 0; i < k && j < my_size; j = my_segment_size) {
            if(j + my_segment_size >= my_size) my_segment_size = my_size - j;
            // destructors are supposed to not throw any exceptions
            destroy( old.table[i].load<relaxed>().pointer<void>(), my_segment_size );
            my_segment_size = i? segment_size( ++i ) : segment_size( i = first_block );
        }
    }
    // free unnecessary segments allocated by reserve() call
    if ( k_stop < k_end ) {
        old.first_block = first_block;
        std::copy(segment_table+k_stop, segment_table+k_end, old.table+k_stop );
        std::fill_n(segment_table+k_stop, (k_end-k_stop), segment_t());
        if( !k ) my_first_block = 0;
    }
    return table;
}

void concurrent_vector_base_v3::internal_swap(concurrent_vector_base_v3& v)
{
    size_type my_sz = my_early_size.load<acquire>();
    size_type v_sz = v.my_early_size.load<relaxed>();
    if(!my_sz && !v_sz) return;

    bool my_was_short = (my_segment.load<relaxed>() == my_storage);
    bool v_was_short  = (v.my_segment.load<relaxed>() == v.my_storage);

    //In C++11, this would be: swap(my_storage, v.my_storage);
    for (int i=0; i < pointers_per_short_table; ++i){
        swap(my_storage[i], v.my_storage[i]);
    }
    tbb::internal::swap<relaxed>(my_first_block, v.my_first_block);
    tbb::internal::swap<relaxed>(my_segment, v.my_segment);
    if (my_was_short){
        v.my_segment.store<relaxed>(v.my_storage);
    }
    if(v_was_short){
        my_segment.store<relaxed>(my_storage);
    }

    my_early_size.store<relaxed>(v_sz);
    v.my_early_size.store<release>(my_sz);
}

} // namespace internal

} // tbb
