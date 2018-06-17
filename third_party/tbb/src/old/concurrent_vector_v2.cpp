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
#include "tbb/tbb_machine.h"
#include "../tbb/itt_notify.h"
#include "tbb/task.h"

#include <stdexcept> // std::length_error
#include <cstring>

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

namespace tbb {

namespace internal {

void concurrent_vector_base::internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op1 init ) {
    size_type e = my_early_size;
    while( e<new_size ) {
        size_type f = my_early_size.compare_and_swap(new_size,e);
        if( f==e ) {
            internal_grow( e, new_size, element_size, init );
            return;
        }
        e = f;
    }
}

class concurrent_vector_base::helper {
    static void extend_segment( concurrent_vector_base& v );
public:
    static segment_index_t find_segment_end( const concurrent_vector_base& v ) {
        const size_t pointers_per_long_segment = sizeof(void*)==4 ? 32 : 64;
        const size_t pointers_per_short_segment = 2;
        //unsigned u = v.my_segment==v.my_storage ? pointers_per_short_segment : pointers_per_long_segment;
        segment_index_t u = v.my_segment==(&(v.my_storage[0])) ? pointers_per_short_segment : pointers_per_long_segment;
        segment_index_t k = 0;
        while( k<u && v.my_segment[k].array )
            ++k;
        return k;
    }
    static void extend_segment_if_necessary( concurrent_vector_base& v, size_t k ) {
        const size_t pointers_per_short_segment = 2;
        if( k>=pointers_per_short_segment && v.my_segment==v.my_storage ) {
            extend_segment(v);
        }
    }
};

void concurrent_vector_base::helper::extend_segment( concurrent_vector_base& v ) {
    const size_t pointers_per_long_segment = sizeof(void*)==4 ? 32 : 64;
    segment_t* s = (segment_t*)NFS_Allocate( pointers_per_long_segment, sizeof(segment_t), NULL );
    std::memset( s, 0, pointers_per_long_segment*sizeof(segment_t) );
    // If other threads are trying to set pointers in the short segment, wait for them to finish their
    // assignments before we copy the short segment to the long segment.
    atomic_backoff backoff;
    while( !v.my_storage[0].array || !v.my_storage[1].array ) backoff.pause();
    s[0] = v.my_storage[0];
    s[1] = v.my_storage[1];
    if( v.my_segment.compare_and_swap( s, v.my_storage )!=v.my_storage )
        NFS_Free(s);
}

concurrent_vector_base::size_type concurrent_vector_base::internal_capacity() const {
    return segment_base( helper::find_segment_end(*this) );
}

void concurrent_vector_base::internal_reserve( size_type n, size_type element_size, size_type max_size ) {
    if( n>max_size ) {
        __TBB_THROW( std::length_error("argument to concurrent_vector::reserve exceeds concurrent_vector::max_size()") );
    }
    for( segment_index_t k = helper::find_segment_end(*this); segment_base(k)<n; ++k ) {
        helper::extend_segment_if_necessary(*this,k);
        size_t m = segment_size(k);
        __TBB_ASSERT( !my_segment[k].array, "concurrent operation during reserve(...)?" );
        my_segment[k].array = NFS_Allocate( m, element_size, NULL );
    }
}

void concurrent_vector_base::internal_copy( const concurrent_vector_base& src, size_type element_size, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    my_early_size = n;
    my_segment = my_storage;
    if( n ) {
        size_type b;
        for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
            helper::extend_segment_if_necessary(*this,k);
            size_t m = segment_size(k);
            __TBB_ASSERT( !my_segment[k].array, "concurrent operation during copy construction?" );
            my_segment[k].array = NFS_Allocate( m, element_size, NULL );
            if( m>n-b ) m = n-b;
            copy( my_segment[k].array, src.my_segment[k].array, m );
        }
    }
}

void concurrent_vector_base::internal_assign( const concurrent_vector_base& src, size_type element_size, internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    while( my_early_size>n ) {
        segment_index_t k = segment_index_of( my_early_size-1 );
        size_type b=segment_base(k);
        size_type new_end = b>=n ? b : n;
        __TBB_ASSERT( my_early_size>new_end, NULL );
        destroy( (char*)my_segment[k].array+element_size*(new_end-b), my_early_size-new_end );
        my_early_size = new_end;
    }
    size_type dst_initialized_size = my_early_size;
    my_early_size = n;
    size_type b;
    for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
        helper::extend_segment_if_necessary(*this,k);
        size_t m = segment_size(k);
        if( !my_segment[k].array )
            my_segment[k].array = NFS_Allocate( m, element_size, NULL );
        if( m>n-b ) m = n-b; 
        size_type a = 0;
        if( dst_initialized_size>b ) {
            a = dst_initialized_size-b;
            if( a>m ) a = m;
            assign( my_segment[k].array, src.my_segment[k].array, a );
            m -= a;
            a *= element_size;
        }
        if( m>0 )
            copy( (char*)my_segment[k].array+a, (char*)src.my_segment[k].array+a, m );
    }
    __TBB_ASSERT( src.my_early_size==n, "detected use of concurrent_vector::operator= with right side that was concurrently modified" );
}

void* concurrent_vector_base::internal_push_back( size_type element_size, size_type& index ) {
    __TBB_ASSERT( sizeof(my_early_size)==sizeof(reference_count), NULL );
    //size_t tmp = __TBB_FetchAndIncrementWacquire(*(tbb::internal::reference_count*)&my_early_size);
    size_t tmp = __TBB_FetchAndIncrementWacquire((tbb::internal::reference_count*)&my_early_size);
    index = tmp;
    segment_index_t k_old = segment_index_of( tmp );
    size_type base = segment_base(k_old);
    helper::extend_segment_if_necessary(*this,k_old);
    segment_t& s = my_segment[k_old];
    void* array = s.array;
    if( !array ) {
        // FIXME - consider factoring this out and share with internal_grow_by
	if( base==tmp ) {
	    __TBB_ASSERT( !s.array, NULL );
            size_t n = segment_size(k_old);
	    array = NFS_Allocate( n, element_size, NULL );
	    ITT_NOTIFY( sync_releasing, &s.array );
	    s.array = array;
	} else {
	    ITT_NOTIFY(sync_prepare, &s.array);
	    spin_wait_while_eq( s.array, (void*)0 );
	    ITT_NOTIFY(sync_acquired, &s.array);
	    array = s.array;
	}
    }
    size_type j_begin = tmp-base;
    return (void*)((char*)array+element_size*j_begin);
}

concurrent_vector_base::size_type concurrent_vector_base::internal_grow_by( size_type delta, size_type element_size, internal_array_op1 init ) {
    size_type result = my_early_size.fetch_and_add(delta);
    internal_grow( result, result+delta, element_size, init );
    return result;
}

void concurrent_vector_base::internal_grow( const size_type start, size_type finish, size_type element_size, internal_array_op1 init ) {
    __TBB_ASSERT( start<finish, "start must be less than finish" );
    size_t tmp = start;
    do {
        segment_index_t k_old = segment_index_of( tmp );
        size_type base = segment_base(k_old);
        size_t n = segment_size(k_old);
        helper::extend_segment_if_necessary(*this,k_old);
        segment_t& s = my_segment[k_old];
        void* array = s.array;
        if( !array ) {
            if( base==tmp ) {
                __TBB_ASSERT( !s.array, NULL );
                array = NFS_Allocate( n, element_size, NULL );
                ITT_NOTIFY( sync_releasing, &s.array );
                s.array = array;
            } else {
                ITT_NOTIFY(sync_prepare, &s.array);
                spin_wait_while_eq( s.array, (void*)0 );
                ITT_NOTIFY(sync_acquired, &s.array);
                array = s.array;
            }
        }
        size_type j_begin = tmp-base;
        size_type j_end = n > finish-base ? finish-base : n;
        (*init)( (void*)((char*)array+element_size*j_begin), j_end-j_begin );
        tmp = base+j_end;
    } while( tmp<finish );
}

void concurrent_vector_base::internal_clear( internal_array_op1 destroy, bool reclaim_storage ) {
    // Set "my_early_size" early, so that subscripting errors can be caught.
    // FIXME - doing so may be hurting exception safety
    __TBB_ASSERT( my_segment, NULL );
    size_type finish = my_early_size;
    my_early_size = 0;
    while( finish>0 ) {
        segment_index_t k_old = segment_index_of(finish-1);
        segment_t& s = my_segment[k_old];
        __TBB_ASSERT( s.array, NULL );
        size_type base = segment_base(k_old);
        size_type j_end = finish-base;
        __TBB_ASSERT( j_end, NULL );
        (*destroy)( s.array, j_end );
        finish = base;
    }

    // Free the arrays
    if( reclaim_storage ) {
        size_t k = helper::find_segment_end(*this);
        while( k>0 ) {
            --k;
            segment_t& s = my_segment[k];
            void* array = s.array;
            s.array = NULL;
            NFS_Free( array );
        }
        // Clear short segment.
        my_storage[0].array = NULL;
        my_storage[1].array = NULL;
        segment_t* s = my_segment;
        if( s!=my_storage ) {
            my_segment = my_storage;
            NFS_Free( s );
        }
    }
}

} // namespace internal

} // tbb
