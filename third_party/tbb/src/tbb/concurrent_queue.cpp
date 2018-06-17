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

#include "tbb/tbb_stddef.h"
#include "tbb/tbb_machine.h"
#include "tbb/tbb_exception.h"
// Define required to satisfy test in internal file.
#define  __TBB_concurrent_queue_H
#include "tbb/internal/_concurrent_queue_impl.h"
#include "concurrent_monitor.h"
#include "itt_notify.h"
#include <new>
#include <cstring>   // for memset()

using namespace std;

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

#define RECORD_EVENTS 0


namespace tbb {

namespace internal {

typedef concurrent_queue_base_v3 concurrent_queue_base;

typedef size_t ticket;

//! A queue using simple locking.
/** For efficiency, this class has no constructor.
    The caller is expected to zero-initialize it. */
struct micro_queue {
    typedef concurrent_queue_base::page page;

    friend class micro_queue_pop_finalizer;

    atomic<page*> head_page;
    atomic<ticket> head_counter;

    atomic<page*> tail_page;
    atomic<ticket> tail_counter;

    spin_mutex page_mutex;

    void push( const void* item, ticket k, concurrent_queue_base& base,
               concurrent_queue_base::copy_specifics op_type );

    void abort_push( ticket k, concurrent_queue_base& base );

    bool pop( void* dst, ticket k, concurrent_queue_base& base );

    micro_queue& assign( const micro_queue& src, concurrent_queue_base& base,
                         concurrent_queue_base::copy_specifics op_type );

    page* make_copy ( concurrent_queue_base& base, const page* src_page, size_t begin_in_page,
                      size_t end_in_page, ticket& g_index, concurrent_queue_base::copy_specifics op_type ) ;

    void make_invalid( ticket k );
};

// we need to yank it out of micro_queue because of concurrent_queue_base::deallocate_page being virtual.
class micro_queue_pop_finalizer: no_copy {
    typedef concurrent_queue_base::page page;
    ticket my_ticket;
    micro_queue& my_queue;
    page* my_page;
    concurrent_queue_base &base;
public:
    micro_queue_pop_finalizer( micro_queue& queue, concurrent_queue_base& b, ticket k, page* p ) :
        my_ticket(k), my_queue(queue), my_page(p), base(b)
    {}
    ~micro_queue_pop_finalizer() {
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
           base.deallocate_page( p );
    }
};

struct predicate_leq {
    ticket t;
    predicate_leq( ticket t_ ) : t(t_) {}
    bool operator() ( uintptr_t p ) const {return (ticket)p<=t;}
};

//! Internal representation of a ConcurrentQueue.
/** For efficiency, this class has no constructor.
    The caller is expected to zero-initialize it. */
class concurrent_queue_rep {
public:
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
    concurrent_monitor items_avail;
    atomic<size_t> n_invalid_entries;
    char pad1[NFS_MaxLineSize-((sizeof(atomic<ticket>)+sizeof(concurrent_monitor)+sizeof(atomic<size_t>))&(NFS_MaxLineSize-1))];

    atomic<ticket> tail_counter;
    concurrent_monitor slots_avail;
    char pad2[NFS_MaxLineSize-((sizeof(atomic<ticket>)+sizeof(concurrent_monitor))&(NFS_MaxLineSize-1))];
    micro_queue array[n_queue];

    micro_queue& choose( ticket k ) {
        // The formula here approximates LRU in a cache-oblivious way.
        return array[index(k)];
    }

    atomic<unsigned> abort_counter;

    //! Value for effective_capacity that denotes unbounded queue.
    static const ptrdiff_t infinite_capacity = ptrdiff_t(~size_t(0)/2);
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // unary minus operator applied to unsigned type, result still unsigned
    #pragma warning( push )
    #pragma warning( disable: 4146 )
#endif

static void* static_invalid_page;

//------------------------------------------------------------------------
// micro_queue
//------------------------------------------------------------------------
void micro_queue::push( const void* item, ticket k, concurrent_queue_base& base,
                        concurrent_queue_base::copy_specifics op_type ) {
    k &= -concurrent_queue_rep::n_queue;
    page* p = NULL;
    // find index on page where we would put the data
    size_t index = modulo_power_of_two( k/concurrent_queue_rep::n_queue, base.items_per_page );
    if( !index ) {  // make a new page
        __TBB_TRY {
            p = base.allocate_page();
        } __TBB_CATCH(...) {
            ++base.my_rep->n_invalid_entries;
            make_invalid( k );
            __TBB_RETHROW();
        }
        p->mask = 0;
        p->next = NULL;
    }

    // wait for my turn
    if( tail_counter!=k ) // The developer insisted on keeping first check out of the backoff loop
        for( atomic_backoff b(true);;b.pause() ) {
            ticket tail = tail_counter;
            if( tail==k ) break;
            else if( tail&0x1 ) {
                // no memory. throws an exception; assumes concurrent_queue_rep::n_queue>1
                ++base.my_rep->n_invalid_entries;
                throw_exception( eid_bad_last_alloc );
            }
        }

    if( p ) { // page is newly allocated; insert in micro_queue
        spin_mutex::scoped_lock lock( page_mutex );
        if( page* q = tail_page )
            q->next = p;
        else
            head_page = p;
        tail_page = p;
    }

    if (item) {
        p = tail_page;
        ITT_NOTIFY( sync_acquired, p );
        __TBB_TRY {
            if( concurrent_queue_base::copy == op_type ) {
                base.copy_item( *p, index, item );
            } else {
                __TBB_ASSERT( concurrent_queue_base::move == op_type, NULL );
                static_cast<concurrent_queue_base_v8&>(base).move_item( *p, index, item );
            }
        }  __TBB_CATCH(...) {
            ++base.my_rep->n_invalid_entries;
            tail_counter += concurrent_queue_rep::n_queue;
            __TBB_RETHROW();
        }
        ITT_NOTIFY( sync_releasing, p );
        // If no exception was thrown, mark item as present.
        p->mask |= uintptr_t(1)<<index;
    }
    else // no item; this was called from abort_push
        ++base.my_rep->n_invalid_entries;

    tail_counter += concurrent_queue_rep::n_queue;
}


void micro_queue::abort_push( ticket k, concurrent_queue_base& base ) {
    push(NULL, k, base, concurrent_queue_base::copy);
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
        micro_queue_pop_finalizer finalizer( *this, base, k+concurrent_queue_rep::n_queue, index==base.items_per_page-1 ? p : NULL );
        if( p->mask & uintptr_t(1)<<index ) {
            success = true;
            ITT_NOTIFY( sync_acquired, dst );
            ITT_NOTIFY( sync_acquired, head_page );
            base.assign_and_destroy_item( dst, *p, index );
            ITT_NOTIFY( sync_releasing, head_page );
        } else {
            --base.my_rep->n_invalid_entries;
        }
    }
    return success;
}

micro_queue& micro_queue::assign( const micro_queue& src, concurrent_queue_base& base,
                                  concurrent_queue_base::copy_specifics op_type )
{
    head_counter = src.head_counter;
    tail_counter = src.tail_counter;

    const page* srcp = src.head_page;
    if( srcp ) {
        ticket g_index = head_counter;
        __TBB_TRY {
            size_t n_items  = (tail_counter-head_counter)/concurrent_queue_rep::n_queue;
            size_t index = modulo_power_of_two( head_counter/concurrent_queue_rep::n_queue, base.items_per_page );
            size_t end_in_first_page = (index+n_items<base.items_per_page)?(index+n_items):base.items_per_page;

            head_page = make_copy( base, srcp, index, end_in_first_page, g_index, op_type );
            page* cur_page = head_page;

            if( srcp != src.tail_page ) {
                for( srcp = srcp->next; srcp!=src.tail_page; srcp=srcp->next ) {
                    cur_page->next = make_copy( base, srcp, 0, base.items_per_page, g_index, op_type );
                    cur_page = cur_page->next;
                }

                __TBB_ASSERT( srcp==src.tail_page, NULL );

                size_t last_index = modulo_power_of_two( tail_counter/concurrent_queue_rep::n_queue, base.items_per_page );
                if( last_index==0 ) last_index = base.items_per_page;

                cur_page->next = make_copy( base, srcp, 0, last_index, g_index, op_type );
                cur_page = cur_page->next;
            }
            tail_page = cur_page;
        } __TBB_CATCH(...) {
            make_invalid( g_index );
            __TBB_RETHROW();
        }
    } else {
        head_page = tail_page = NULL;
    }
    return *this;
}

concurrent_queue_base::page* micro_queue::make_copy( concurrent_queue_base& base,
    const concurrent_queue_base::page* src_page, size_t begin_in_page, size_t end_in_page,
    ticket& g_index, concurrent_queue_base::copy_specifics op_type )
{
    page* new_page = base.allocate_page();
    new_page->next = NULL;
    new_page->mask = src_page->mask;
    for( ; begin_in_page!=end_in_page; ++begin_in_page, ++g_index )
        if( new_page->mask & uintptr_t(1)<<begin_in_page ) {
            if( concurrent_queue_base::copy == op_type ) {
                base.copy_page_item( *new_page, begin_in_page, *src_page, begin_in_page );
            } else {
                __TBB_ASSERT( concurrent_queue_base::move == op_type, NULL );
                static_cast<concurrent_queue_base_v8&>(base).move_page_item( *new_page, begin_in_page, *src_page, begin_in_page );
            }
        }
    return new_page;
}

void micro_queue::make_invalid( ticket k )
{
    static concurrent_queue_base::page dummy = {static_cast<page*>((void*)1), 0};
    // mark it so that no more pushes are allowed.
    static_invalid_page = &dummy;
    {
        spin_mutex::scoped_lock lock( page_mutex );
        tail_counter = k+concurrent_queue_rep::n_queue+1;
        if( page* q = tail_page )
            q->next = static_cast<page*>(static_invalid_page);
        else
            head_page = static_cast<page*>(static_invalid_page);
        tail_page = static_cast<page*>(static_invalid_page);
    }
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning( pop )
#endif // warning 4146 is back

//------------------------------------------------------------------------
// concurrent_queue_base
//------------------------------------------------------------------------
concurrent_queue_base_v3::concurrent_queue_base_v3( size_t item_sz ) {
    items_per_page = item_sz<=  8 ? 32 :
                     item_sz<= 16 ? 16 :
                     item_sz<= 32 ?  8 :
                     item_sz<= 64 ?  4 :
                     item_sz<=128 ?  2 :
                     1;
    my_capacity = size_t(-1)/(item_sz>1 ? item_sz : 2);
    my_rep = cache_aligned_allocator<concurrent_queue_rep>().allocate(1);
    __TBB_ASSERT( is_aligned(my_rep, NFS_GetLineSize()), "alignment error" );
    __TBB_ASSERT( is_aligned(&my_rep->head_counter, NFS_GetLineSize()), "alignment error" );
    __TBB_ASSERT( is_aligned(&my_rep->tail_counter, NFS_GetLineSize()), "alignment error" );
    __TBB_ASSERT( is_aligned(&my_rep->array, NFS_GetLineSize()), "alignment error" );
    memset(my_rep,0,sizeof(concurrent_queue_rep));
    new ( &my_rep->items_avail ) concurrent_monitor();
    new ( &my_rep->slots_avail ) concurrent_monitor();
    this->item_size = item_sz;
}

concurrent_queue_base_v3::~concurrent_queue_base_v3() {
    size_t nq = my_rep->n_queue;
    for( size_t i=0; i<nq; i++ )
        __TBB_ASSERT( my_rep->array[i].tail_page==NULL, "pages were not freed properly" );
    cache_aligned_allocator<concurrent_queue_rep>().deallocate(my_rep,1);
}

void concurrent_queue_base_v3::internal_push( const void* src ) {
    internal_insert_item( src, copy );
}

void concurrent_queue_base_v8::internal_push_move( const void* src ) {
   internal_insert_item( src, move );
}

void concurrent_queue_base_v3::internal_insert_item( const void* src, copy_specifics op_type ) {
    concurrent_queue_rep& r = *my_rep;
    unsigned old_abort_counter = r.abort_counter;
    ticket k = r.tail_counter++;
    ptrdiff_t e = my_capacity;
#if DO_ITT_NOTIFY
    bool sync_prepare_done = false;
#endif
    if( (ptrdiff_t)(k-r.head_counter)>=e ) { // queue is full
#if DO_ITT_NOTIFY
        if( !sync_prepare_done ) {
            ITT_NOTIFY( sync_prepare, &sync_prepare_done );
            sync_prepare_done = true;
        }
#endif
        bool slept = false;
        concurrent_monitor::thread_context thr_ctx;
        r.slots_avail.prepare_wait( thr_ctx, ((ptrdiff_t)(k-e)) );
        while( (ptrdiff_t)(k-r.head_counter)>=const_cast<volatile ptrdiff_t&>(e = my_capacity) ) {
            __TBB_TRY {
                if( r.abort_counter!=old_abort_counter ) {
                    r.slots_avail.cancel_wait( thr_ctx );
                    throw_exception( eid_user_abort );
                }
                slept = r.slots_avail.commit_wait( thr_ctx );
            } __TBB_CATCH( tbb::user_abort& ) {
                r.choose(k).abort_push(k, *this);
                __TBB_RETHROW();
            } __TBB_CATCH(...) {
                __TBB_RETHROW();
            }
            if (slept == true) break;
            r.slots_avail.prepare_wait( thr_ctx, ((ptrdiff_t)(k-e)) );
        }
        if( !slept )
            r.slots_avail.cancel_wait( thr_ctx );
    }
    ITT_NOTIFY( sync_acquired, &sync_prepare_done );
    __TBB_ASSERT( (ptrdiff_t)(k-r.head_counter)<my_capacity, NULL);
    r.choose( k ).push( src, k, *this, op_type );
    r.items_avail.notify( predicate_leq(k) );
}

void concurrent_queue_base_v3::internal_pop( void* dst ) {
    concurrent_queue_rep& r = *my_rep;
    ticket k;
#if DO_ITT_NOTIFY
    bool sync_prepare_done = false;
#endif
    unsigned old_abort_counter = r.abort_counter;
    // This loop is a single pop operation; abort_counter should not be re-read inside
    do {
        k=r.head_counter++;
        if ( (ptrdiff_t)(r.tail_counter-k)<=0 ) { // queue is empty
#if DO_ITT_NOTIFY
            if( !sync_prepare_done ) {
                ITT_NOTIFY( sync_prepare, dst );
                sync_prepare_done = true;
            }
#endif
            bool slept = false;
            concurrent_monitor::thread_context thr_ctx;
            r.items_avail.prepare_wait( thr_ctx, k );
            while( (ptrdiff_t)(r.tail_counter-k)<=0 ) {
                __TBB_TRY {
                    if( r.abort_counter!=old_abort_counter ) {
                        r.items_avail.cancel_wait( thr_ctx );
                        throw_exception( eid_user_abort );
                    }
                    slept = r.items_avail.commit_wait( thr_ctx );
                } __TBB_CATCH( tbb::user_abort& ) {
                    r.head_counter--;
                    __TBB_RETHROW();
                } __TBB_CATCH(...) {
                    __TBB_RETHROW();
                }
                if (slept == true) break;
                r.items_avail.prepare_wait( thr_ctx, k );
            }
            if( !slept )
                r.items_avail.cancel_wait( thr_ctx );
        }
        __TBB_ASSERT((ptrdiff_t)(r.tail_counter-k)>0, NULL);
    } while( !r.choose(k).pop(dst,k,*this) );

    // wake up a producer..
    r.slots_avail.notify( predicate_leq(k) );
}

void concurrent_queue_base_v3::internal_abort() {
    concurrent_queue_rep& r = *my_rep;
    ++r.abort_counter;
    r.items_avail.abort_all();
    r.slots_avail.abort_all();
}

bool concurrent_queue_base_v3::internal_pop_if_present( void* dst ) {
    concurrent_queue_rep& r = *my_rep;
    ticket k;
    do {
        k = r.head_counter;
        for(;;) {
            if( (ptrdiff_t)(r.tail_counter-k)<=0 ) {
                // Queue is empty
                return false;
            }
            // Queue had item with ticket k when we looked.  Attempt to get that item.
            ticket tk=k;
            k = r.head_counter.compare_and_swap( tk+1, tk );
            if( k==tk )
                break;
            // Another thread snatched the item, retry.
        }
    } while( !r.choose( k ).pop( dst, k, *this ) );

    r.slots_avail.notify( predicate_leq(k) );

    return true;
}

bool concurrent_queue_base_v3::internal_push_if_not_full( const void* src ) {
    return internal_insert_if_not_full( src, copy );
}

bool concurrent_queue_base_v8::internal_push_move_if_not_full( const void* src ) {
    return internal_insert_if_not_full( src, move );
}

bool concurrent_queue_base_v3::internal_insert_if_not_full( const void* src, copy_specifics op_type ) {
    concurrent_queue_rep& r = *my_rep;
    ticket k = r.tail_counter;
    for(;;) {
        if( (ptrdiff_t)(k-r.head_counter)>=my_capacity ) {
            // Queue is full
            return false;
        }
        // Queue had empty slot with ticket k when we looked.  Attempt to claim that slot.
        ticket tk=k;
        k = r.tail_counter.compare_and_swap( tk+1, tk );
        if( k==tk )
            break;
        // Another thread claimed the slot, so retry.
    }
    r.choose(k).push(src, k, *this, op_type);
    r.items_avail.notify( predicate_leq(k) );
    return true;
}

ptrdiff_t concurrent_queue_base_v3::internal_size() const {
    __TBB_ASSERT( sizeof(ptrdiff_t)<=sizeof(size_t), NULL );
    return ptrdiff_t(my_rep->tail_counter-my_rep->head_counter-my_rep->n_invalid_entries);
}

bool concurrent_queue_base_v3::internal_empty() const {
    ticket tc = my_rep->tail_counter;
    ticket hc = my_rep->head_counter;
    // if tc!=r.tail_counter, the queue was not empty at some point between the two reads.
    return ( tc==my_rep->tail_counter && ptrdiff_t(tc-hc-my_rep->n_invalid_entries)<=0 );
}

void concurrent_queue_base_v3::internal_set_capacity( ptrdiff_t capacity, size_t /*item_sz*/ ) {
    my_capacity = capacity<0 ? concurrent_queue_rep::infinite_capacity : capacity;
}

void concurrent_queue_base_v3::internal_finish_clear() {
    size_t nq = my_rep->n_queue;
    for( size_t i=0; i<nq; ++i ) {
        page* tp = my_rep->array[i].tail_page;
        __TBB_ASSERT( my_rep->array[i].head_page==tp, "at most one page should remain" );
        if( tp!=NULL) {
            if( tp!=static_invalid_page ) deallocate_page( tp );
            my_rep->array[i].tail_page = NULL;
        }
    }
}

void concurrent_queue_base_v3::internal_throw_exception() const {
    throw_exception( eid_bad_alloc );
}

void concurrent_queue_base_v3::internal_assign( const concurrent_queue_base& src, copy_specifics op_type ) {
    items_per_page = src.items_per_page;
    my_capacity = src.my_capacity;

    // copy concurrent_queue_rep.
    my_rep->head_counter = src.my_rep->head_counter;
    my_rep->tail_counter = src.my_rep->tail_counter;
    my_rep->n_invalid_entries = src.my_rep->n_invalid_entries;
    my_rep->abort_counter = src.my_rep->abort_counter;

    // copy micro_queues
    for( size_t i = 0; i<my_rep->n_queue; ++i )
        my_rep->array[i].assign( src.my_rep->array[i], *this, op_type );

    __TBB_ASSERT( my_rep->head_counter==src.my_rep->head_counter && my_rep->tail_counter==src.my_rep->tail_counter,
            "the source concurrent queue should not be concurrently modified." );
}

void concurrent_queue_base_v3::assign( const concurrent_queue_base& src ) {
    internal_assign( src, copy );
}

void concurrent_queue_base_v8::move_content( concurrent_queue_base_v8& src ) {
    internal_assign( src, move );
}

//------------------------------------------------------------------------
// concurrent_queue_iterator_rep
//------------------------------------------------------------------------
class concurrent_queue_iterator_rep: no_assign {
public:
    ticket head_counter;
    const concurrent_queue_base& my_queue;
    const size_t offset_of_last;
    concurrent_queue_base::page* array[concurrent_queue_rep::n_queue];
    concurrent_queue_iterator_rep( const concurrent_queue_base& queue, size_t offset_of_last_ ) :
        head_counter(queue.my_rep->head_counter),
        my_queue(queue),
        offset_of_last(offset_of_last_)
    {
        const concurrent_queue_rep& rep = *queue.my_rep;
        for( size_t k=0; k<concurrent_queue_rep::n_queue; ++k )
            array[k] = rep.array[k].head_page;
    }
    //! Set item to point to kth element.  Return true if at end of queue or item is marked valid; false otherwise.
    bool get_item( void*& item, size_t k ) {
        if( k==my_queue.my_rep->tail_counter ) {
            item = NULL;
            return true;
        } else {
            concurrent_queue_base::page* p = array[concurrent_queue_rep::index(k)];
            __TBB_ASSERT(p,NULL);
            size_t i = modulo_power_of_two( k/concurrent_queue_rep::n_queue, my_queue.items_per_page );
            item = static_cast<unsigned char*>(static_cast<void*>(p)) + offset_of_last + my_queue.item_size*i;
            return (p->mask & uintptr_t(1)<<i)!=0;
        }
    }
};

//------------------------------------------------------------------------
// concurrent_queue_iterator_base
//------------------------------------------------------------------------

void concurrent_queue_iterator_base_v3::initialize( const concurrent_queue_base& queue, size_t offset_of_last ) {
    my_rep = cache_aligned_allocator<concurrent_queue_iterator_rep>().allocate(1);
    new( my_rep ) concurrent_queue_iterator_rep(queue,offset_of_last);
    size_t k = my_rep->head_counter;
    if( !my_rep->get_item(my_item, k) ) advance();
}

concurrent_queue_iterator_base_v3::concurrent_queue_iterator_base_v3( const concurrent_queue_base& queue ) {
    initialize(queue,0);
}

concurrent_queue_iterator_base_v3::concurrent_queue_iterator_base_v3( const concurrent_queue_base& queue, size_t offset_of_last ) {
    initialize(queue,offset_of_last);
}

void concurrent_queue_iterator_base_v3::assign( const concurrent_queue_iterator_base& other ) {
    if( my_rep!=other.my_rep ) {
        if( my_rep ) {
            cache_aligned_allocator<concurrent_queue_iterator_rep>().deallocate(my_rep, 1);
            my_rep = NULL;
        }
        if( other.my_rep ) {
            my_rep = cache_aligned_allocator<concurrent_queue_iterator_rep>().allocate(1);
            new( my_rep ) concurrent_queue_iterator_rep( *other.my_rep );
        }
    }
    my_item = other.my_item;
}

void concurrent_queue_iterator_base_v3::advance() {
    __TBB_ASSERT( my_item, "attempt to increment iterator past end of queue" );
    size_t k = my_rep->head_counter;
    const concurrent_queue_base& queue = my_rep->my_queue;
#if TBB_USE_ASSERT
    void* tmp;
    my_rep->get_item(tmp,k);
    __TBB_ASSERT( my_item==tmp, NULL );
#endif /* TBB_USE_ASSERT */
    size_t i = modulo_power_of_two( k/concurrent_queue_rep::n_queue, queue.items_per_page );
    if( i==queue.items_per_page-1 ) {
        concurrent_queue_base::page*& root = my_rep->array[concurrent_queue_rep::index(k)];
        root = root->next;
    }
    // advance k
    my_rep->head_counter = ++k;
    if( !my_rep->get_item(my_item, k) ) advance();
}

concurrent_queue_iterator_base_v3::~concurrent_queue_iterator_base_v3() {
    //delete my_rep;
    cache_aligned_allocator<concurrent_queue_iterator_rep>().deallocate(my_rep, 1);
    my_rep = NULL;
}

} // namespace internal

} // namespace tbb
