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

#include "concurrent_queue_v2.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/spin_mutex.h"
#include "tbb/atomic.h"
#include <cstring>
#include <stdio.h>

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

#define RECORD_EVENTS 0

using namespace std;

namespace tbb {

namespace internal {

class concurrent_queue_rep;

//! A queue using simple locking.
/** For efficiency, this class has no constructor.
    The caller is expected to zero-initialize it. */
struct micro_queue {
    typedef concurrent_queue_base::page page;
    typedef size_t ticket;

    atomic<page*> head_page;
    atomic<ticket> head_counter;

    atomic<page*> tail_page;
    atomic<ticket> tail_counter;

    spin_mutex page_mutex;

    class push_finalizer: no_copy {
        ticket my_ticket;
        micro_queue& my_queue;
    public:
        push_finalizer( micro_queue& queue, ticket k ) :
            my_ticket(k), my_queue(queue)
        {}
        ~push_finalizer() {
            my_queue.tail_counter = my_ticket;
        }
    };

    void push( const void* item, ticket k, concurrent_queue_base& base );

    class pop_finalizer: no_copy {
        ticket my_ticket;
        micro_queue& my_queue;
        page* my_page;
    public:
        pop_finalizer( micro_queue& queue, ticket k, page* p ) :
            my_ticket(k), my_queue(queue), my_page(p)
        {}
        ~pop_finalizer() {
            page* p = my_page;
            if( p ) {
                spin_mutex::scoped_lock lock( my_queue.page_mutex );
                page* q = p->next;
                my_queue.head_page = q;
                if( !q ) {
                    my_queue.tail_page = NULL;
                }
            }
            my_queue.head_counter = my_ticket;
            if( p )
                operator delete(p);
        }
    };

    bool pop( void* dst, ticket k, concurrent_queue_base& base );
};

//! Internal representation of a ConcurrentQueue.
/** For efficiency, this class has no constructor.
    The caller is expected to zero-initialize it. */
class concurrent_queue_rep {
public:
    typedef size_t ticket;

private:
    friend struct micro_queue;

    //! Approximately n_queue/golden ratio
    static const size_t phi = 3;

public:
    //! Must be power of 2
    static const size_t n_queue = 8;

    //! Map ticket to an array index
    static size_t index( ticket k ) {
        return k*phi%n_queue;
    }

    atomic<ticket> head_counter;
    char pad1[NFS_MaxLineSize-sizeof(atomic<ticket>)];

    atomic<ticket> tail_counter;
    char pad2[NFS_MaxLineSize-sizeof(atomic<ticket>)];
    micro_queue array[n_queue];

    micro_queue& choose( ticket k ) {
        // The formula here approximates LRU in a cache-oblivious way.
        return array[index(k)];
    }

    //! Value for effective_capacity that denotes unbounded queue.
    static const ptrdiff_t infinite_capacity = ptrdiff_t(~size_t(0)/2);
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // unary minus operator applied to unsigned type, result still unsigned
    #pragma warning( push )
    #pragma warning( disable: 4146 )
#endif

//------------------------------------------------------------------------
// micro_queue
//------------------------------------------------------------------------
void micro_queue::push( const void* item, ticket k, concurrent_queue_base& base ) {
    k &= -concurrent_queue_rep::n_queue;
    page* p = NULL;
    size_t index = modulo_power_of_two( k/concurrent_queue_rep::n_queue, base.items_per_page );
    if( !index ) {
        size_t n = sizeof(page) + base.items_per_page*base.item_size;
        p = static_cast<page*>(operator new( n ));
        p->mask = 0;
        p->next = NULL;
    }
    {
        push_finalizer finalizer( *this, k+concurrent_queue_rep::n_queue );
        spin_wait_until_eq( tail_counter, k );
        if( p ) {
            spin_mutex::scoped_lock lock( page_mutex );
            if( page* q = tail_page )
                q->next = p;
            else
                head_page = p;
            tail_page = p;
        } else {
            p = tail_page;
        }
        base.copy_item( *p, index, item );
        // If no exception was thrown, mark item as present.
        p->mask |= uintptr_t(1)<<index;
    }
}

bool micro_queue::pop( void* dst, ticket k, concurrent_queue_base& base ) {
    k &= -concurrent_queue_rep::n_queue;
    spin_wait_until_eq( head_counter, k );
    spin_wait_while_eq( tail_counter, k );
    page *p = head_page;
    __TBB_ASSERT( p, NULL );
    size_t index = modulo_power_of_two( k/concurrent_queue_rep::n_queue, base.items_per_page );
    bool success = false;
    {
        pop_finalizer finalizer( *this, k+concurrent_queue_rep::n_queue, index==base.items_per_page-1 ? p : NULL );
        if( p->mask & uintptr_t(1)<<index ) {
            success = true;
            base.assign_and_destroy_item( dst, *p, index );
        }
    }
    return success;
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning( pop )
#endif

//------------------------------------------------------------------------
// concurrent_queue_base
//------------------------------------------------------------------------
concurrent_queue_base::concurrent_queue_base( size_t item_sz ) {
    items_per_page = item_sz<=  8 ? 32 :
                     item_sz<= 16 ? 16 :
                     item_sz<= 32 ?  8 :
                     item_sz<= 64 ?  4 :
                     item_sz<=128 ?  2 :
                     1;
    my_capacity = size_t(-1)/(item_sz>1 ? item_sz : 2);
    my_rep = cache_aligned_allocator<concurrent_queue_rep>().allocate(1);
    __TBB_ASSERT( (size_t)my_rep % NFS_GetLineSize()==0, "alignment error" );
    __TBB_ASSERT( (size_t)&my_rep->head_counter % NFS_GetLineSize()==0, "alignment error" );
    __TBB_ASSERT( (size_t)&my_rep->tail_counter % NFS_GetLineSize()==0, "alignment error" );
    __TBB_ASSERT( (size_t)&my_rep->array % NFS_GetLineSize()==0, "alignment error" );
    memset(my_rep,0,sizeof(concurrent_queue_rep));
    this->item_size = item_sz;
}

concurrent_queue_base::~concurrent_queue_base() {
    size_t nq = my_rep->n_queue;
    for( size_t i=0; i<nq; i++ ) {
        page* tp = my_rep->array[i].tail_page;
        __TBB_ASSERT( my_rep->array[i].head_page==tp, "at most one page should remain" );
        if( tp!=NULL )
            delete tp;
    }
    cache_aligned_allocator<concurrent_queue_rep>().deallocate(my_rep,1);
}

void concurrent_queue_base::internal_push( const void* src ) {
    concurrent_queue_rep& r = *my_rep;
    concurrent_queue_rep::ticket k  = r.tail_counter++;
    if( my_capacity<concurrent_queue_rep::infinite_capacity ) {
        // Capacity is limited, wait to not exceed it
        atomic_backoff backoff;
        while( (ptrdiff_t)(k-r.head_counter)>=const_cast<volatile ptrdiff_t&>(my_capacity) )
            backoff.pause();
    }
    r.choose(k).push(src,k,*this);
}

void concurrent_queue_base::internal_pop( void* dst ) {
    concurrent_queue_rep& r = *my_rep;
    concurrent_queue_rep::ticket k;
    do {
        k = r.head_counter++;
    } while( !r.choose(k).pop(dst,k,*this) );
}

bool concurrent_queue_base::internal_pop_if_present( void* dst ) {
    concurrent_queue_rep& r = *my_rep;
    concurrent_queue_rep::ticket k;
    do {
        for( atomic_backoff b;;b.pause() ) {
            k = r.head_counter;
            if( r.tail_counter<=k ) {
                // Queue is empty
                return false;
            }
            // Queue had item with ticket k when we looked.  Attempt to get that item.
            if( r.head_counter.compare_and_swap(k+1,k)==k ) {
                break;
            }
            // Another thread snatched the item, so pause and retry.
        }
    } while( !r.choose(k).pop(dst,k,*this) );
    return true;
}

bool concurrent_queue_base::internal_push_if_not_full( const void* src ) {
    concurrent_queue_rep& r = *my_rep;
    concurrent_queue_rep::ticket k;
    for( atomic_backoff b;;b.pause() ) {
        k = r.tail_counter;
        if( (ptrdiff_t)(k-r.head_counter)>=my_capacity ) {
            // Queue is full
            return false;
        }
        // Queue had empty slot with ticket k when we looked.  Attempt to claim that slot.
        if( r.tail_counter.compare_and_swap(k+1,k)==k )
            break;
        // Another thread claimed the slot, so pause and retry.
    }
    r.choose(k).push(src,k,*this);
    return true;
}

ptrdiff_t concurrent_queue_base::internal_size() const {
    __TBB_ASSERT( sizeof(ptrdiff_t)<=sizeof(size_t), NULL );
    return ptrdiff_t(my_rep->tail_counter-my_rep->head_counter);
}

void concurrent_queue_base::internal_set_capacity( ptrdiff_t capacity, size_t /*item_sz*/ ) {
    my_capacity = capacity<0 ? concurrent_queue_rep::infinite_capacity : capacity;
}

//------------------------------------------------------------------------
// concurrent_queue_iterator_rep
//------------------------------------------------------------------------
class  concurrent_queue_iterator_rep: no_assign {
public:
    typedef concurrent_queue_rep::ticket ticket;
    ticket head_counter;
    const concurrent_queue_base& my_queue;
    concurrent_queue_base::page* array[concurrent_queue_rep::n_queue];
    concurrent_queue_iterator_rep( const concurrent_queue_base& queue ) :
        head_counter(queue.my_rep->head_counter),
        my_queue(queue)
    {
        const concurrent_queue_rep& rep = *queue.my_rep;
        for( size_t k=0; k<concurrent_queue_rep::n_queue; ++k )
            array[k] = rep.array[k].head_page;
    }
    //! Get pointer to kth element
    void* choose( size_t k ) {
        if( k==my_queue.my_rep->tail_counter )
            return NULL;
        else {
            concurrent_queue_base::page* p = array[concurrent_queue_rep::index(k)];
            __TBB_ASSERT(p,NULL);
            size_t i = modulo_power_of_two( k/concurrent_queue_rep::n_queue, my_queue.items_per_page );
            return static_cast<unsigned char*>(static_cast<void*>(p+1)) + my_queue.item_size*i;
        }
    }
};

//------------------------------------------------------------------------
// concurrent_queue_iterator_base
//------------------------------------------------------------------------
concurrent_queue_iterator_base::concurrent_queue_iterator_base( const concurrent_queue_base& queue ) {
    my_rep = new concurrent_queue_iterator_rep(queue);
    my_item = my_rep->choose(my_rep->head_counter);
}

void concurrent_queue_iterator_base::assign( const concurrent_queue_iterator_base& other ) {
    if( my_rep!=other.my_rep ) {
        if( my_rep ) {
            delete my_rep;
            my_rep = NULL;
        }
        if( other.my_rep ) {
            my_rep = new concurrent_queue_iterator_rep( *other.my_rep );
        }
    }
    my_item = other.my_item;
}

void concurrent_queue_iterator_base::advance() {
    __TBB_ASSERT( my_item, "attempt to increment iterator past end of queue" );
    size_t k = my_rep->head_counter;
    const concurrent_queue_base& queue = my_rep->my_queue;
    __TBB_ASSERT( my_item==my_rep->choose(k), NULL );
    size_t i = modulo_power_of_two( k/concurrent_queue_rep::n_queue, queue.items_per_page );
    if( i==queue.items_per_page-1 ) {
        concurrent_queue_base::page*& root = my_rep->array[concurrent_queue_rep::index(k)];
        root = root->next;
    }
    my_rep->head_counter = k+1;
    my_item = my_rep->choose(k+1);
}

concurrent_queue_iterator_base::~concurrent_queue_iterator_base() {
    delete my_rep;
    my_rep = NULL;
}

} // namespace internal

} // namespace tbb
