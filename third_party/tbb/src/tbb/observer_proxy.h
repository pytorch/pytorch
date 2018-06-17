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

#ifndef _TBB_observer_proxy_H
#define _TBB_observer_proxy_H

#if __TBB_SCHEDULER_OBSERVER

#include "scheduler_common.h" // to include task.h
#include "tbb/task_scheduler_observer.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/aligned_space.h"

namespace tbb {
namespace internal {

class observer_list {
    friend class arena;

    // Mutex is wrapped with aligned_space to shut up warnings when its destructor
    // is called while threads are still using it.
    typedef aligned_space<spin_rw_mutex>  my_mutex_type;

    //! Pointer to the head of this list.
    observer_proxy* my_head;

    //! Pointer to the tail of this list.
    observer_proxy* my_tail;

    //! Mutex protecting this list.
    my_mutex_type my_mutex;

    //! Back-pointer to the arena this list belongs to.
    arena* my_arena;

    //! Decrement refcount of the proxy p if there are other outstanding references.
    /** In case of success sets p to NULL. Must be invoked from under the list lock. **/
    inline static void remove_ref_fast( observer_proxy*& p );

    //! Implements notify_entry_observers functionality.
    void do_notify_entry_observers( observer_proxy*& last, bool worker );

    //! Implements notify_exit_observers functionality.
    void do_notify_exit_observers( observer_proxy* last, bool worker );

public:
    observer_list () : my_head(NULL), my_tail(NULL) {}

    //! Removes and destroys all observer proxies from the list.
    /** Cannot be used concurrently with other methods. **/
    void clear ();

    //! Add observer proxy to the tail of the list.
    void insert ( observer_proxy* p );

    //! Remove observer proxy from the list.
    void remove ( observer_proxy* p );

    //! Decrement refcount of the proxy and destroy it if necessary.
    /** When refcount reaches zero removes the proxy from the list and destructs it. **/
    void remove_ref( observer_proxy* p );

    //! Type of the scoped lock for the reader-writer mutex associated with the list.
    typedef spin_rw_mutex::scoped_lock scoped_lock;

    //! Accessor to the reader-writer mutex associated with the list.
    spin_rw_mutex& mutex () { return my_mutex.begin()[0]; }

    bool empty () const { return my_head == NULL; }

    //! Call entry notifications on observers added after last was notified.
    /** Updates last to become the last notified observer proxy (in the global list)
        or leaves it to be NULL. The proxy has its refcount incremented. **/
    inline void notify_entry_observers( observer_proxy*& last, bool worker );

    //! Call exit notifications on last and observers added before it.
    inline void notify_exit_observers( observer_proxy*& last, bool worker );

    //! Call may_sleep callbacks to ask for permission for a worker thread to leave market
    bool ask_permission_to_leave();
}; // class observer_list

//! Wrapper for an observer object
/** To maintain shared lists of observers the scheduler first wraps each observer
    object into a proxy so that a list item remained valid even after the corresponding
    proxy object is destroyed by the user code. **/
class observer_proxy {
    friend class task_scheduler_observer_v3;
    friend class observer_list;
    //! Reference count used for garbage collection.
    /** 1 for reference from my task_scheduler_observer.
        1 for each task dispatcher's last observer pointer.
        No accounting for neighbors in the shared list. */
    atomic<int> my_ref_count;
    //! Reference to the list this observer belongs to.
    observer_list* my_list;
    //! Pointer to next observer in the list specified by my_head.
    /** NULL for the last item in the list. **/
    observer_proxy* my_next;
    //! Pointer to the previous observer in the list specified by my_head.
    /** For the head of the list points to the last item. **/
    observer_proxy* my_prev;
    //! Associated observer
    task_scheduler_observer_v3* my_observer;
    //! Version
    char my_version;

#if __TBB_ARENA_OBSERVER || __TBB_SLEEP_PERMISSION
    interface6::task_scheduler_observer* get_v6_observer();
#endif
#if __TBB_ARENA_OBSERVER
    bool is_global(); //TODO: move them back inline when un-CPF'ing
#endif

    //! Constructs proxy for the given observer and adds it to the specified list.
    observer_proxy( task_scheduler_observer_v3& );

#if TBB_USE_ASSERT
    ~observer_proxy();
#endif /* TBB_USE_ASSERT */

    //! Shut up the warning
    observer_proxy& operator = ( const observer_proxy& );
}; // class observer_proxy

inline void observer_list::remove_ref_fast( observer_proxy*& p ) {
    if( p->my_observer ) {
        // Can decrement refcount quickly, as it cannot drop to zero while under the lock.
        int r = --p->my_ref_count;
        __TBB_ASSERT_EX( r, NULL );
        p = NULL;
    } else {
        // Use slow form of refcount decrementing, after the lock is released.
    }
}

inline void observer_list::notify_entry_observers( observer_proxy*& last, bool worker ) {
    if ( last == my_tail )
        return;
    do_notify_entry_observers( last, worker );
}

inline void observer_list::notify_exit_observers( observer_proxy*& last, bool worker ) {
    if ( !last )
        return;
    __TBB_ASSERT(is_alive((uintptr_t)last), NULL);
    do_notify_exit_observers( last, worker );
    __TBB_ASSERT(last, NULL);
    poison_value(last);
}

extern padded<observer_list> the_global_observer_list;

} // namespace internal
} // namespace tbb

#endif /* __TBB_SCHEDULER_OBSERVER */

#endif /* _TBB_observer_proxy_H */
