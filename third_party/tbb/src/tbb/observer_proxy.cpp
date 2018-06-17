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

#include "tbb/tbb_config.h"

#if __TBB_SCHEDULER_OBSERVER

#include "observer_proxy.h"
#include "tbb_main.h"
#include "governor.h"
#include "scheduler.h"
#include "arena.h"

namespace tbb {
namespace internal {

padded<observer_list> the_global_observer_list;

#if TBB_USE_ASSERT
static atomic<int> observer_proxy_count;

struct check_observer_proxy_count {
    ~check_observer_proxy_count() {
        if( observer_proxy_count!=0 ) {
            runtime_warning( "Leaked %ld observer_proxy objects\n", long(observer_proxy_count) );
        }
    }
};

static check_observer_proxy_count the_check_observer_proxy_count;
#endif /* TBB_USE_ASSERT */

#if __TBB_ARENA_OBSERVER || __TBB_SLEEP_PERMISSION
interface6::task_scheduler_observer* observer_proxy::get_v6_observer() {
    if(my_version != 6) return NULL;
    return static_cast<interface6::task_scheduler_observer*>(my_observer);
}
#endif

#if __TBB_ARENA_OBSERVER
bool observer_proxy::is_global() {
    return !get_v6_observer() || get_v6_observer()->my_context_tag == interface6::task_scheduler_observer::global_tag;
}
#endif /* __TBB_ARENA_OBSERVER */

observer_proxy::observer_proxy( task_scheduler_observer_v3& tso )
    : my_list(NULL), my_next(NULL), my_prev(NULL), my_observer(&tso)
{
#if TBB_USE_ASSERT
    ++observer_proxy_count;
#endif /* TBB_USE_ASSERT */
    // 1 for observer
    my_ref_count = 1;
    my_version =
#if __TBB_ARENA_OBSERVER
        load<relaxed>(my_observer->my_busy_count)
                 == interface6::task_scheduler_observer::v6_trait ? 6 :
#endif
        0;
    __TBB_ASSERT( my_version >= 6 || !load<relaxed>(my_observer->my_busy_count), NULL );
}

#if TBB_USE_ASSERT
observer_proxy::~observer_proxy () {
    __TBB_ASSERT( !my_ref_count, "Attempt to destroy proxy still in use" );
    poison_value(my_ref_count);
    poison_pointer(my_prev);
    poison_pointer(my_next);
    --observer_proxy_count;
}
#endif /* TBB_USE_ASSERT */

template<memory_semantics M, class T, class V>
T atomic_fetch_and_store ( T* addr, const V& val ) {
    return (T)atomic_traits<sizeof(T), M>::fetch_and_store( addr, (T)val );
}

void observer_list::clear () {
    __TBB_ASSERT( this != &the_global_observer_list, "Method clear() cannot be used on the list of global observers" );
    // Though the method will work fine for the empty list, we require the caller
    // to check for the list emptiness before invoking it to avoid extra overhead.
    __TBB_ASSERT( !empty(), NULL );
    {
        scoped_lock lock(mutex(), /*is_writer=*/true);
        observer_proxy *next = my_head;
        while ( observer_proxy *p = next ) {
            __TBB_ASSERT( p->my_version >= 6, NULL );
            next = p->my_next;
            // Both proxy p and observer p->my_observer (if non-null) are guaranteed
            // to be alive while the list is locked.
            task_scheduler_observer_v3 *obs = p->my_observer;
            // Make sure that possible concurrent observer destruction does not
            // conflict with the proxy list cleanup.
            if ( !obs || !(p = (observer_proxy*)__TBB_FetchAndStoreW(&obs->my_proxy, 0)) )
                continue;
            // accessing 'obs' after detaching of obs->my_proxy leads to the race with observer destruction
            __TBB_ASSERT( !next || p == next->my_prev, NULL );
            __TBB_ASSERT( is_alive(p->my_ref_count), "Observer's proxy died prematurely" );
            __TBB_ASSERT( p->my_ref_count == 1, "Reference for observer is missing" );
#if TBB_USE_ASSERT
            p->my_observer = NULL;
            p->my_ref_count = 0;
#endif /* TBB_USE_ASSERT */
            remove(p);
            delete p;
        }
    }
    while( my_head )
        __TBB_Yield();
}

void observer_list::insert ( observer_proxy* p ) {
    scoped_lock lock(mutex(), /*is_writer=*/true);
    if ( my_head ) {
        p->my_prev = my_tail;
        my_tail->my_next = p;
    }
    else
        my_head = p;
    my_tail = p;
}

void observer_list::remove ( observer_proxy* p ) {
    __TBB_ASSERT( my_head, "Attempt to remove an item from an empty list" );
    __TBB_ASSERT( !my_tail->my_next, "Last item's my_next must be NULL" );
    if( p == my_tail ) {
        __TBB_ASSERT( !p->my_next, NULL );
        my_tail = p->my_prev;
    }
    else {
        __TBB_ASSERT( p->my_next, NULL );
        p->my_next->my_prev = p->my_prev;
    }
    if ( p == my_head ) {
        __TBB_ASSERT( !p->my_prev, NULL );
        my_head = p->my_next;
    }
    else {
        __TBB_ASSERT( p->my_prev, NULL );
        p->my_prev->my_next = p->my_next;
    }
    __TBB_ASSERT( (my_head && my_tail) || (!my_head && !my_tail), NULL );
}

void observer_list::remove_ref( observer_proxy* p ) {
    int r = p->my_ref_count;
    __TBB_ASSERT( is_alive(r), NULL );
    while(r>1) {
        __TBB_ASSERT( r!=0, NULL );
        int r_old = p->my_ref_count.compare_and_swap(r-1,r);
        if( r_old==r ) {
            // Successfully decremented count.
            return;
        }
        r = r_old;
    }
    __TBB_ASSERT( r==1, NULL );
    // Reference count might go to zero
    {
        // Use lock to avoid resurrection by a thread concurrently walking the list
        observer_list::scoped_lock lock(mutex(), /*is_writer=*/true);
        r = --p->my_ref_count;
        if( !r )
            remove(p);
    }
    __TBB_ASSERT( r || !p->my_ref_count, NULL );
    if( !r )
        delete p;
}

void observer_list::do_notify_entry_observers( observer_proxy*& last, bool worker ) {
    // Pointer p marches though the list from last (exclusively) to the end.
    observer_proxy *p = last, *prev = p;
    for(;;) {
        task_scheduler_observer_v3* tso=NULL;
        // Hold lock on list only long enough to advance to the next proxy in the list.
        {
            scoped_lock lock(mutex(), /*is_writer=*/false);
            do {
                if( p ) {
                    // We were already processing the list.
                    if( observer_proxy* q = p->my_next ) {
                        if( p == prev )
                            remove_ref_fast(prev); // sets prev to NULL if successful
                        p = q;
                    }
                    else {
                        // Reached the end of the list.
                        if( p == prev ) {
                            // Keep the reference as we store the 'last' pointer in scheduler
                            __TBB_ASSERT(p->my_ref_count >= 1 + (p->my_observer?1:0), NULL);
                        } else {
                            // The last few proxies were empty
                            __TBB_ASSERT(p->my_ref_count, NULL);
                            ++p->my_ref_count;
                            if( prev ) {
                                lock.release();
                                remove_ref(prev);
                            }
                        }
                        last = p;
                        return;
                    }
                } else {
                    // Starting pass through the list
                    p = my_head;
                    if( !p )
                        return;
                }
                tso = p->my_observer;
            } while( !tso );
            ++p->my_ref_count;
            ++tso->my_busy_count;
        }
        __TBB_ASSERT( !prev || p!=prev, NULL );
        // Release the proxy pinned before p
        if( prev )
            remove_ref(prev);
        // Do not hold any locks on the list while calling user's code.
        // Do not intercept any exceptions that may escape the callback so that
        // they are either handled by the TBB scheduler or passed to the debugger.
        tso->on_scheduler_entry(worker);
        __TBB_ASSERT(p->my_ref_count, NULL);
        intptr_t bc = --tso->my_busy_count;
        __TBB_ASSERT_EX( bc>=0, "my_busy_count underflowed" );
        prev = p;
    }
}

void observer_list::do_notify_exit_observers( observer_proxy* last, bool worker ) {
    // Pointer p marches though the list from the beginning to last (inclusively).
    observer_proxy *p = NULL, *prev = NULL;
    for(;;) {
        task_scheduler_observer_v3* tso=NULL;
        // Hold lock on list only long enough to advance to the next proxy in the list.
        {
            scoped_lock lock(mutex(), /*is_writer=*/false);
            do {
                if( p ) {
                    // We were already processing the list.
                    if( p != last ) {
                        __TBB_ASSERT( p->my_next, "List items before 'last' must have valid my_next pointer" );
                        if( p == prev )
                            remove_ref_fast(prev); // sets prev to NULL if successful
                        p = p->my_next;
                    } else {
                        // remove the reference from the last item
                        remove_ref_fast(p);
                        if( p ) {
                            lock.release();
                            remove_ref(p);
                        }
                        return;
                    }
                } else {
                    // Starting pass through the list
                    p = my_head;
                    __TBB_ASSERT( p, "Nonzero 'last' must guarantee that the global list is non-empty" );
                }
                tso = p->my_observer;
            } while( !tso );
            // The item is already refcounted
            if ( p != last ) // the last is already referenced since entry notification
                ++p->my_ref_count;
            ++tso->my_busy_count;
        }
        __TBB_ASSERT( !prev || p!=prev, NULL );
        if( prev )
            remove_ref(prev);
        // Do not hold any locks on the list while calling user's code.
        // Do not intercept any exceptions that may escape the callback so that
        // they are either handled by the TBB scheduler or passed to the debugger.
        tso->on_scheduler_exit(worker);
        __TBB_ASSERT(p->my_ref_count || p == last, NULL);
        intptr_t bc = --tso->my_busy_count;
        __TBB_ASSERT_EX( bc>=0, "my_busy_count underflowed" );
        prev = p;
    }
}

#if __TBB_SLEEP_PERMISSION
bool observer_list::ask_permission_to_leave() {
    __TBB_ASSERT( this == &the_global_observer_list, "This method cannot be used on lists of arena observers" );
    if( !my_head ) return true;
    // Pointer p marches though the list
    observer_proxy *p = NULL, *prev = NULL;
    bool result = true;
    while( result ) {
        task_scheduler_observer* tso = NULL;
        // Hold lock on list only long enough to advance to the next proxy in the list.
        {
            scoped_lock lock(mutex(), /*is_writer=*/false);
            do {
                if( p ) {
                    // We were already processing the list.
                    observer_proxy* q = p->my_next;
                    // read next, remove the previous reference
                    if( p == prev )
                        remove_ref_fast(prev); // sets prev to NULL if successful
                    if( q ) p = q;
                    else {
                        // Reached the end of the list.
                        if( prev ) {
                            lock.release();
                            remove_ref(prev);
                        }
                        return result;
                    }
                } else {
                    // Starting pass through the list
                    p = my_head;
                    if( !p )
                        return result;
                }
                tso = p->get_v6_observer();
            } while( !tso );
            ++p->my_ref_count;
            ++tso->my_busy_count;
        }
        __TBB_ASSERT( !prev || p!=prev, NULL );
        // Release the proxy pinned before p
        if( prev )
            remove_ref(prev);
        // Do not hold any locks on the list while calling user's code.
        // Do not intercept any exceptions that may escape the callback so that
        // they are either handled by the TBB scheduler or passed to the debugger.
        result = tso->may_sleep();
        __TBB_ASSERT(p->my_ref_count, NULL);
        intptr_t bc = --tso->my_busy_count;
        __TBB_ASSERT_EX( bc>=0, "my_busy_count underflowed" );
        prev = p;
    }
    if( prev )
        remove_ref(prev);
    return result;
}
#endif//__TBB_SLEEP_PERMISSION

void task_scheduler_observer_v3::observe( bool enable ) {
    if( enable ) {
        if( !my_proxy ) {
            my_proxy = new observer_proxy( *this );
            my_busy_count = 0; // proxy stores versioning information, clear it
#if __TBB_ARENA_OBSERVER
            if ( !my_proxy->is_global() ) {
                // Local observer activation
                generic_scheduler* s = governor::local_scheduler_if_initialized();
                __TBB_ASSERT( my_proxy->get_v6_observer(), NULL );
                intptr_t tag = my_proxy->get_v6_observer()->my_context_tag;
                if( tag != interface6::task_scheduler_observer::implicit_tag ) { // explicit arena
                    task_arena *a = reinterpret_cast<task_arena*>(tag);
                    a->initialize();
                    my_proxy->my_list = &a->my_arena->my_observers;
                } else {
                    if( !s )
                        s = governor::init_scheduler( task_scheduler_init::automatic, 0, true );
                    __TBB_ASSERT( __TBB_InitOnce::initialization_done(), NULL );
                    __TBB_ASSERT( s && s->my_arena, NULL );
                    my_proxy->my_list = &s->my_arena->my_observers;
                }
                my_proxy->my_list->insert(my_proxy);
                // Notify newly activated observer and other pending ones if it belongs to current arena
                if(s && &s->my_arena->my_observers == my_proxy->my_list )
                    my_proxy->my_list->notify_entry_observers( s->my_last_local_observer, s->is_worker() );
            } else
#endif /* __TBB_ARENA_OBSERVER */
            {
                // Obsolete. Global observer activation
                if( !__TBB_InitOnce::initialization_done() )
                    DoOneTimeInitializations();
                my_proxy->my_list = &the_global_observer_list;
                my_proxy->my_list->insert(my_proxy);
                if( generic_scheduler* s = governor::local_scheduler_if_initialized() ) {
                    // Notify newly created observer of its own thread.
                    // Any other pending observers are notified too.
                    the_global_observer_list.notify_entry_observers( s->my_last_global_observer, s->is_worker() );
                }
            }
        }
    } else {
        // Make sure that possible concurrent proxy list cleanup does not conflict
        // with the observer destruction here.
        if ( observer_proxy* proxy = (observer_proxy*)__TBB_FetchAndStoreW(&my_proxy, 0) ) {
            // List destruction should not touch this proxy after we've won the above interlocked exchange.
            __TBB_ASSERT( proxy->my_observer == this, NULL );
            __TBB_ASSERT( is_alive(proxy->my_ref_count), "Observer's proxy died prematurely" );
            __TBB_ASSERT( proxy->my_ref_count >= 1, "reference for observer missing" );
            observer_list &list = *proxy->my_list;
            {
                // Ensure that none of the list walkers relies on observer pointer validity
                observer_list::scoped_lock lock(list.mutex(), /*is_writer=*/true);
                proxy->my_observer = NULL;
                // Proxy may still be held by other threads (to track the last notified observer)
                if( !--proxy->my_ref_count ) {// nobody can increase it under exclusive lock
                    list.remove(proxy);
                    __TBB_ASSERT( !proxy->my_ref_count, NULL );
                    delete proxy;
                }
            }
            while( my_busy_count ) // other threads are still accessing the callback
                __TBB_Yield();
        }
    }
}

} // namespace internal
} // namespace tbb

#endif /* __TBB_SCHEDULER_OBSERVER */
