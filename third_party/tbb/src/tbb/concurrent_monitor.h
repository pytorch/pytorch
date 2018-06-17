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

#ifndef __TBB_concurrent_monitor_H
#define __TBB_concurrent_monitor_H

#include "tbb/tbb_stddef.h"
#include "tbb/atomic.h"
#include "tbb/spin_mutex.h"
#include "tbb/tbb_exception.h"
#include "tbb/aligned_space.h"

#include "semaphore.h"

namespace tbb {
namespace internal {

//! Circular doubly-linked list with sentinel
/** head.next points to the front and head.prev points to the back */
class circular_doubly_linked_list_with_sentinel : no_copy {
public:
    struct node_t {
        node_t* next;
        node_t* prev;
        explicit node_t() : next((node_t*)(uintptr_t)0xcdcdcdcd), prev((node_t*)(uintptr_t)0xcdcdcdcd) {}
    };

    // ctor
    circular_doubly_linked_list_with_sentinel() {clear();}
    // dtor
    ~circular_doubly_linked_list_with_sentinel() {__TBB_ASSERT( head.next==&head && head.prev==&head, "the list is not empty" );}

    inline size_t  size()  const {return count;}
    inline bool    empty() const {return size()==0;}
    inline node_t* front() const {return head.next;}
    inline node_t* last()  const {return head.prev;}
    inline node_t* begin() const {return front();}
    inline const node_t* end() const {return &head;}

    //! add to the back of the list
    inline void add( node_t* n ) {
        __TBB_store_relaxed(count, __TBB_load_relaxed(count) + 1);
        n->prev = head.prev;
        n->next = &head;
        head.prev->next = n;
        head.prev = n;
    }

    //! remove node 'n'
    inline void remove( node_t& n ) {
        __TBB_ASSERT( count > 0, "attempt to remove an item from an empty list" );
        __TBB_store_relaxed(count, __TBB_load_relaxed(count) - 1);
        n.prev->next = n.next;
        n.next->prev = n.prev;
    }

    //! move all elements to 'lst' and initialize the 'this' list
    inline void flush_to( circular_doubly_linked_list_with_sentinel& lst ) {
        if( const size_t l_count = __TBB_load_relaxed(count) ) {
            __TBB_store_relaxed(lst.count, l_count);
            lst.head.next = head.next;
            lst.head.prev = head.prev;
            head.next->prev = &lst.head;
            head.prev->next = &lst.head;
            clear();
        }
    }

    void clear() {head.next = head.prev = &head; __TBB_store_relaxed(count, 0);}
private:
    __TBB_atomic size_t count;
    node_t head;
};

typedef circular_doubly_linked_list_with_sentinel waitset_t;
typedef circular_doubly_linked_list_with_sentinel::node_t waitset_node_t;

//! concurrent_monitor
/** fine-grained concurrent_monitor implementation */
class concurrent_monitor : no_copy {
public:
    /** per-thread descriptor for concurrent_monitor */
    class thread_context : waitset_node_t, no_copy {
        friend class concurrent_monitor;
    public:
        thread_context() : spurious(false), aborted(false), ready(false), context(0) {
            epoch = 0;
            in_waitset = false;
        }
        ~thread_context() {
            if (ready) {
                if( spurious ) semaphore().P();
                semaphore().~binary_semaphore();
            }
        }
        binary_semaphore& semaphore() { return *sema.begin(); }
    private:
        //! The method for lazy initialization of the thread_context's semaphore.
        //  Inlining of the method is undesirable, due to extra instructions for
        //  exception support added at caller side.
        __TBB_NOINLINE( void init() );
        tbb::aligned_space<binary_semaphore> sema;
        __TBB_atomic unsigned epoch;
        tbb::atomic<bool> in_waitset;
        bool  spurious;
        bool  aborted;
        bool  ready;
        uintptr_t context;
    };

    //! ctor
    concurrent_monitor() {__TBB_store_relaxed(epoch, 0);}

    //! dtor
    ~concurrent_monitor() ;

    //! prepare wait by inserting 'thr' into the wait queue
    void prepare_wait( thread_context& thr, uintptr_t ctx = 0 );

    //! Commit wait if event count has not changed; otherwise, cancel wait.
    /** Returns true if committed, false if canceled. */
    inline bool commit_wait( thread_context& thr ) {
        const bool do_it = thr.epoch == __TBB_load_relaxed(epoch);
        // this check is just an optimization
        if( do_it ) {
            __TBB_ASSERT( thr.ready, "use of commit_wait() without prior prepare_wait()");
            thr.semaphore().P();
            __TBB_ASSERT( !thr.in_waitset, "still in the queue?" );
            if( thr.aborted )
                throw_exception( eid_user_abort );
        } else {
            cancel_wait( thr );
        }
        return do_it;
    }
    //! Cancel the wait. Removes the thread from the wait queue if not removed yet.
    void cancel_wait( thread_context& thr );

    //! Wait for a condition to be satisfied with waiting-on context
    template<typename WaitUntil, typename Context>
    void wait( WaitUntil until, Context on );

    //! Notify one thread about the event
    void notify_one() {atomic_fence(); notify_one_relaxed();}

    //! Notify one thread about the event. Relaxed version.
    void notify_one_relaxed();

    //! Notify all waiting threads of the event
    void notify_all() {atomic_fence(); notify_all_relaxed();}

    //! Notify all waiting threads of the event; Relaxed version
    void notify_all_relaxed();

    //! Notify waiting threads of the event that satisfies the given predicate
    template<typename P> void notify( const P& predicate ) {atomic_fence(); notify_relaxed( predicate );}

    //! Notify waiting threads of the event that satisfies the given predicate; Relaxed version
    template<typename P> void notify_relaxed( const P& predicate );

    //! Abort any sleeping threads at the time of the call
    void abort_all() {atomic_fence(); abort_all_relaxed(); }

    //! Abort any sleeping threads at the time of the call; Relaxed version
    void abort_all_relaxed();

private:
    tbb::spin_mutex mutex_ec;
    waitset_t       waitset_ec;
    __TBB_atomic unsigned epoch;
    thread_context* to_thread_context( waitset_node_t* n ) { return static_cast<thread_context*>(n); }
};

template<typename WaitUntil, typename Context>
void concurrent_monitor::wait( WaitUntil until, Context on )
{
    bool slept = false;
    thread_context thr_ctx;
    prepare_wait( thr_ctx, on() );
    while( !until() ) {
        if( (slept = commit_wait( thr_ctx ) )==true )
            if( until() ) break;
        slept = false;
        prepare_wait( thr_ctx, on() );
    }
    if( !slept )
        cancel_wait( thr_ctx );
}

template<typename P>
void concurrent_monitor::notify_relaxed( const P& predicate ) {
        if( waitset_ec.empty() )
            return;
        waitset_t temp;
        waitset_node_t* nxt;
        const waitset_node_t* end = waitset_ec.end();
        {
            tbb::spin_mutex::scoped_lock l( mutex_ec );
            __TBB_store_relaxed(epoch, __TBB_load_relaxed(epoch) + 1);
            for( waitset_node_t* n=waitset_ec.last(); n!=end; n=nxt ) {
                nxt = n->prev;
                thread_context* thr = to_thread_context( n );
                if( predicate( thr->context ) ) {
                    waitset_ec.remove( *n );
                    thr->in_waitset = false;
                    temp.add( n );
                }
            }
        }

        end = temp.end();
        for( waitset_node_t* n=temp.front(); n!=end; n=nxt ) {
            nxt = n->next;
            to_thread_context(n)->semaphore().V();
        }
#if TBB_USE_ASSERT
        temp.clear();
#endif
}

} // namespace internal
} // namespace tbb

#endif /* __TBB_concurrent_monitor_H */
