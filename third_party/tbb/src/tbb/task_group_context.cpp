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

#include "scheduler.h"

#include "itt_notify.h"

namespace tbb {

#if __TBB_TASK_GROUP_CONTEXT

using namespace internal;

//------------------------------------------------------------------------
// captured_exception
//------------------------------------------------------------------------

inline char* duplicate_string ( const char* src ) {
    char* dst = NULL;
    if ( src ) {
        size_t len = strlen(src) + 1;
        dst = (char*)allocate_via_handler_v3(len);
        strncpy (dst, src, len);
    }
    return dst;
}

captured_exception::~captured_exception () throw() {
    clear();
}

void captured_exception::set ( const char* a_name, const char* info ) throw() {
    my_exception_name = duplicate_string( a_name );
    my_exception_info = duplicate_string( info );
}

void captured_exception::clear () throw() {
    deallocate_via_handler_v3 (const_cast<char*>(my_exception_name));
    deallocate_via_handler_v3 (const_cast<char*>(my_exception_info));
}

captured_exception* captured_exception::move () throw() {
    captured_exception *e = (captured_exception*)allocate_via_handler_v3(sizeof(captured_exception));
    if ( e ) {
        ::new (e) captured_exception();
        e->my_exception_name = my_exception_name;
        e->my_exception_info = my_exception_info;
        e->my_dynamic = true;
        my_exception_name = my_exception_info = NULL;
    }
    return e;
}

void captured_exception::destroy () throw() {
    __TBB_ASSERT ( my_dynamic, "Method destroy can be used only on objects created by clone or allocate" );
    if ( my_dynamic ) {
        this->captured_exception::~captured_exception();
        deallocate_via_handler_v3 (this);
    }
}

captured_exception* captured_exception::allocate ( const char* a_name, const char* info ) {
    captured_exception *e = (captured_exception*)allocate_via_handler_v3( sizeof(captured_exception) );
    if ( e ) {
        ::new (e) captured_exception(a_name, info);
        e->my_dynamic = true;
    }
    return e;
}

const char* captured_exception::name() const throw() {
    return my_exception_name;
}

const char* captured_exception::what() const throw() {
    return my_exception_info;
}


//------------------------------------------------------------------------
// tbb_exception_ptr
//------------------------------------------------------------------------

#if !TBB_USE_CAPTURED_EXCEPTION

namespace internal {

template<typename T>
tbb_exception_ptr* AllocateExceptionContainer( const T& src ) {
    tbb_exception_ptr *eptr = (tbb_exception_ptr*)allocate_via_handler_v3( sizeof(tbb_exception_ptr) );
    if ( eptr )
        new (eptr) tbb_exception_ptr(src);
    return eptr;
}

tbb_exception_ptr* tbb_exception_ptr::allocate () {
    return AllocateExceptionContainer( std::current_exception() );
}

tbb_exception_ptr* tbb_exception_ptr::allocate ( const tbb_exception& ) {
    return AllocateExceptionContainer( std::current_exception() );
}

tbb_exception_ptr* tbb_exception_ptr::allocate ( captured_exception& src ) {
    tbb_exception_ptr *res = AllocateExceptionContainer( src );
    src.destroy();
    return res;
}

void tbb_exception_ptr::destroy () throw() {
    this->tbb_exception_ptr::~tbb_exception_ptr();
    deallocate_via_handler_v3 (this);
}

} // namespace internal
#endif /* !TBB_USE_CAPTURED_EXCEPTION */


//------------------------------------------------------------------------
// task_group_context
//------------------------------------------------------------------------

task_group_context::~task_group_context () {
    if ( __TBB_load_relaxed(my_kind) == binding_completed ) {
        if ( governor::is_set(my_owner) ) {
            // Local update of the context list
            uintptr_t local_count_snapshot = my_owner->my_context_state_propagation_epoch;
            my_owner->my_local_ctx_list_update.store<relaxed>(1);
            // Prevent load of nonlocal update flag from being hoisted before the
            // store to local update flag.
            atomic_fence();
            if ( my_owner->my_nonlocal_ctx_list_update.load<relaxed>() ) {
                spin_mutex::scoped_lock lock(my_owner->my_context_list_mutex);
                my_node.my_prev->my_next = my_node.my_next;
                my_node.my_next->my_prev = my_node.my_prev;
                my_owner->my_local_ctx_list_update.store<relaxed>(0);
            }
            else {
                my_node.my_prev->my_next = my_node.my_next;
                my_node.my_next->my_prev = my_node.my_prev;
                // Release fence is necessary so that update of our neighbors in
                // the context list was committed when possible concurrent destroyer
                // proceeds after local update flag is reset by the following store.
                my_owner->my_local_ctx_list_update.store<release>(0);
                if ( local_count_snapshot != the_context_state_propagation_epoch ) {
                    // Another thread was propagating cancellation request when we removed
                    // ourselves from the list. We must ensure that it is not accessing us
                    // when this destructor finishes. We'll be able to acquire the lock
                    // below only after the other thread finishes with us.
                    spin_mutex::scoped_lock lock(my_owner->my_context_list_mutex);
                }
            }
        }
        else {
            // Nonlocal update of the context list
            // Synchronizes with generic_scheduler::cleanup_local_context_list()
            // TODO: evaluate and perhaps relax, or add some lock instead
            if ( internal::as_atomic(my_kind).fetch_and_store(dying) == detached ) {
                my_node.my_prev->my_next = my_node.my_next;
                my_node.my_next->my_prev = my_node.my_prev;
            }
            else {
                //TODO: evaluate and perhaps relax
                my_owner->my_nonlocal_ctx_list_update.fetch_and_increment<full_fence>();
                //TODO: evaluate and perhaps remove
                spin_wait_until_eq( my_owner->my_local_ctx_list_update, 0u );
                my_owner->my_context_list_mutex.lock();
                my_node.my_prev->my_next = my_node.my_next;
                my_node.my_next->my_prev = my_node.my_prev;
                my_owner->my_context_list_mutex.unlock();
                //TODO: evaluate and perhaps relax
                my_owner->my_nonlocal_ctx_list_update.fetch_and_decrement<full_fence>();
            }
        }
    }
#if __TBB_FP_CONTEXT
    internal::punned_cast<cpu_ctl_env*>(&my_cpu_ctl_env)->~cpu_ctl_env();
#endif
    poison_value(my_version_and_traits);
    if ( my_exception )
        my_exception->destroy();
    ITT_STACK(itt_caller != ITT_CALLER_NULL, caller_destroy, itt_caller);
}

void task_group_context::init () {
    __TBB_STATIC_ASSERT ( sizeof(my_version_and_traits) >= 4, "Layout of my_version_and_traits must be reconsidered on this platform" );
    __TBB_STATIC_ASSERT ( sizeof(task_group_context) == 2 * NFS_MaxLineSize, "Context class has wrong size - check padding and members alignment" );
    __TBB_ASSERT ( (uintptr_t(this) & (sizeof(my_cancellation_requested) - 1)) == 0, "Context is improperly aligned" );
    __TBB_ASSERT ( __TBB_load_relaxed(my_kind) == isolated || __TBB_load_relaxed(my_kind) == bound, "Context can be created only as isolated or bound" );
    my_parent = NULL;
    my_cancellation_requested = 0;
    my_exception = NULL;
    my_owner = NULL;
    my_state = 0;
    itt_caller = ITT_CALLER_NULL;
#if __TBB_TASK_PRIORITY
    my_priority = normalized_normal_priority;
#endif /* __TBB_TASK_PRIORITY */
#if __TBB_FP_CONTEXT
    __TBB_STATIC_ASSERT( sizeof(my_cpu_ctl_env) == sizeof(internal::uint64_t), "The reserved space for FPU settings are not equal sizeof(uint64_t)" );
    __TBB_STATIC_ASSERT( sizeof(cpu_ctl_env) <= sizeof(my_cpu_ctl_env), "FPU settings storage does not fit to uint64_t" );
    suppress_unused_warning( my_cpu_ctl_env.space );

    cpu_ctl_env &ctl = *internal::punned_cast<cpu_ctl_env*>(&my_cpu_ctl_env);
    new ( &ctl ) cpu_ctl_env;
    if ( my_version_and_traits & fp_settings )
        ctl.get_env();
#endif
}

void task_group_context::register_with ( generic_scheduler *local_sched ) {
    __TBB_ASSERT( local_sched, NULL );
    my_owner = local_sched;
    // state propagation logic assumes new contexts are bound to head of the list
    my_node.my_prev = &local_sched->my_context_list_head;
    // Notify threads that may be concurrently destroying contexts registered
    // in this scheduler's list that local list update is underway.
    local_sched->my_local_ctx_list_update.store<relaxed>(1);
    // Prevent load of global propagation epoch counter from being hoisted before
    // speculative stores above, as well as load of nonlocal update flag from
    // being hoisted before the store to local update flag.
    atomic_fence();
    // Finalize local context list update
    if ( local_sched->my_nonlocal_ctx_list_update.load<relaxed>() ) {
        spin_mutex::scoped_lock lock(my_owner->my_context_list_mutex);
        local_sched->my_context_list_head.my_next->my_prev = &my_node;
        my_node.my_next = local_sched->my_context_list_head.my_next;
        my_owner->my_local_ctx_list_update.store<relaxed>(0);
        local_sched->my_context_list_head.my_next = &my_node;
    }
    else {
        local_sched->my_context_list_head.my_next->my_prev = &my_node;
        my_node.my_next = local_sched->my_context_list_head.my_next;
        my_owner->my_local_ctx_list_update.store<release>(0);
        // Thread-local list of contexts allows concurrent traversal by another thread
        // while propagating state change. To ensure visibility of my_node's members
        // to the concurrently traversing thread, the list's head is updated by means
        // of store-with-release.
        __TBB_store_with_release(local_sched->my_context_list_head.my_next, &my_node);
    }
}

void task_group_context::bind_to ( generic_scheduler *local_sched ) {
    __TBB_ASSERT ( __TBB_load_relaxed(my_kind) == binding_required, "Already bound or isolated?" );
    __TBB_ASSERT ( !my_parent, "Parent is set before initial binding" );
    my_parent = local_sched->my_innermost_running_task->prefix().context;
#if __TBB_FP_CONTEXT
    // Inherit FPU settings only if the context has not captured FPU settings yet.
    if ( !(my_version_and_traits & fp_settings) )
        copy_fp_settings(*my_parent);
#endif

    // Condition below prevents unnecessary thrashing parent context's cache line
    if ( !(my_parent->my_state & may_have_children) )
        my_parent->my_state |= may_have_children; // full fence is below
    if ( my_parent->my_parent ) {
        // Even if this context were made accessible for state change propagation
        // (by placing __TBB_store_with_release(s->my_context_list_head.my_next, &my_node)
        // above), it still could be missed if state propagation from a grand-ancestor
        // was underway concurrently with binding.
        // Speculative propagation from the parent together with epoch counters
        // detecting possibility of such a race allow to avoid taking locks when
        // there is no contention.

        // Acquire fence is necessary to prevent reordering subsequent speculative
        // loads of parent state data out of the scope where epoch counters comparison
        // can reliably validate it.
        uintptr_t local_count_snapshot = __TBB_load_with_acquire( my_parent->my_owner->my_context_state_propagation_epoch );
        // Speculative propagation of parent's state. The speculation will be
        // validated by the epoch counters check further on.
        my_cancellation_requested = my_parent->my_cancellation_requested;
#if __TBB_TASK_PRIORITY
        my_priority = my_parent->my_priority;
#endif /* __TBB_TASK_PRIORITY */
        register_with( local_sched ); // Issues full fence

        // If no state propagation was detected by the following condition, the above
        // full fence guarantees that the parent had correct state during speculative
        // propagation before the fence. Otherwise the propagation from parent is
        // repeated under the lock.
        if ( local_count_snapshot != the_context_state_propagation_epoch ) {
            // Another thread may be propagating state change right now. So resort to lock.
            context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
            my_cancellation_requested = my_parent->my_cancellation_requested;
#if __TBB_TASK_PRIORITY
            my_priority = my_parent->my_priority;
#endif /* __TBB_TASK_PRIORITY */
        }
    }
    else {
        register_with( local_sched ); // Issues full fence
        // As we do not have grand-ancestors, concurrent state propagation (if any)
        // may originate only from the parent context, and thus it is safe to directly
        // copy the state from it.
        my_cancellation_requested = my_parent->my_cancellation_requested;
#if __TBB_TASK_PRIORITY
        my_priority = my_parent->my_priority;
#endif /* __TBB_TASK_PRIORITY */
    }
    __TBB_store_relaxed(my_kind, binding_completed);
}

template <typename T>
void task_group_context::propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state ) {
    if (this->*mptr_state == new_state) {
        // Nothing to do, whether descending from "src" or not, so no need to scan.
        // Hopefully this happens often thanks to earlier invocations.
        // This optimization is enabled by LIFO order in the context lists:
        // - new contexts are bound to the beginning of lists;
        // - descendants are newer than ancestors;
        // - earlier invocations are therefore likely to "paint" long chains.
    }
    else if (this == &src) {
        // This clause is disjunct from the traversal below, which skips src entirely.
        // Note that src.*mptr_state is not necessarily still equal to new_state (another thread may have changed it again).
        // Such interference is probably not frequent enough to aim for optimisation by writing new_state again (to make the other thread back down).
        // Letting the other thread prevail may also be fairer.
    }
    else {
        for ( task_group_context *ancestor = my_parent; ancestor != NULL; ancestor = ancestor->my_parent ) {
            __TBB_ASSERT(internal::is_alive(ancestor->my_version_and_traits), "context tree was corrupted");
            if ( ancestor == &src ) {
                for ( task_group_context *ctx = this; ctx != ancestor; ctx = ctx->my_parent )
                    ctx->*mptr_state = new_state;
                break;
            }
        }
    }
}

template <typename T>
void generic_scheduler::propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state ) {
    spin_mutex::scoped_lock lock(my_context_list_mutex);
    // Acquire fence is necessary to ensure that the subsequent node->my_next load
    // returned the correct value in case it was just inserted in another thread.
    // The fence also ensures visibility of the correct my_parent value.
    context_list_node_t *node = __TBB_load_with_acquire(my_context_list_head.my_next);
    while ( node != &my_context_list_head ) {
        task_group_context &ctx = __TBB_get_object_ref(task_group_context, my_node, node);
        if ( ctx.*mptr_state != new_state )
            ctx.propagate_task_group_state( mptr_state, src, new_state );
        node = node->my_next;
        __TBB_ASSERT( is_alive(ctx.my_version_and_traits), "Local context list contains destroyed object" );
    }
    // Sync up local propagation epoch with the global one. Release fence prevents
    // reordering of possible store to *mptr_state after the sync point.
    __TBB_store_with_release(my_context_state_propagation_epoch, the_context_state_propagation_epoch);
}

template <typename T>
bool market::propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state ) {
    if ( !(src.my_state & task_group_context::may_have_children) )
        return true;
    // The whole propagation algorithm is under the lock in order to ensure correctness
    // in case of concurrent state changes at the different levels of the context tree.
    // See comment at the bottom of scheduler.cpp
    context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
    if ( src.*mptr_state != new_state )
        // Another thread has concurrently changed the state. Back down.
        return false;
    // Advance global state propagation epoch
    __TBB_FetchAndAddWrelease(&the_context_state_propagation_epoch, 1);
    // Propagate to all workers and masters and sync up their local epochs with the global one
    unsigned num_workers = my_first_unused_worker_idx;
    for ( unsigned i = 0; i < num_workers; ++i ) {
        generic_scheduler *s = my_workers[i];
        // If the worker is only about to be registered, skip it.
        if ( s )
            s->propagate_task_group_state( mptr_state, src, new_state );
    }
    // Propagate to all master threads
    // The whole propagation sequence is locked, thus no contention is expected
    for( scheduler_list_type::iterator it = my_masters.begin(); it != my_masters.end(); it++  )
        it->propagate_task_group_state( mptr_state, src, new_state );
    return true;
}

bool task_group_context::cancel_group_execution () {
    __TBB_ASSERT ( my_cancellation_requested == 0 || my_cancellation_requested == 1, "Invalid cancellation state");
    if ( my_cancellation_requested || as_atomic(my_cancellation_requested).compare_and_swap(1, 0) ) {
        // This task group and any descendants have already been canceled.
        // (A newly added descendant would inherit its parent's my_cancellation_requested,
        // not missing out on any cancellation still being propagated, and a context cannot be uncanceled.)
        return false;
    }
    governor::local_scheduler_weak()->my_market->propagate_task_group_state( &task_group_context::my_cancellation_requested, *this, (uintptr_t)1 );
    return true;
}

bool task_group_context::is_group_execution_cancelled () const {
    return my_cancellation_requested != 0;
}

// IMPORTANT: It is assumed that this method is not used concurrently!
void task_group_context::reset () {
    //! TODO: Add assertion that this context does not have children
    // No fences are necessary since this context can be accessed from another thread
    // only after stealing happened (which means necessary fences were used).
    if ( my_exception )  {
        my_exception->destroy();
        my_exception = NULL;
    }
    my_cancellation_requested = 0;
}

#if __TBB_FP_CONTEXT
// IMPORTANT: It is assumed that this method is not used concurrently!
void task_group_context::capture_fp_settings () {
    //! TODO: Add assertion that this context does not have children
    // No fences are necessary since this context can be accessed from another thread
    // only after stealing happened (which means necessary fences were used).
    cpu_ctl_env &ctl = *internal::punned_cast<cpu_ctl_env*>(&my_cpu_ctl_env);
    if ( !(my_version_and_traits & fp_settings) ) {
        new ( &ctl ) cpu_ctl_env;
        my_version_and_traits |= fp_settings;
    }
    ctl.get_env();
}

void task_group_context::copy_fp_settings( const task_group_context &src ) {
    __TBB_ASSERT( !(my_version_and_traits & fp_settings), "The context already has FPU settings." );
    __TBB_ASSERT( src.my_version_and_traits & fp_settings, "The source context does not have FPU settings." );

    cpu_ctl_env &ctl = *internal::punned_cast<cpu_ctl_env*>(&my_cpu_ctl_env);
    cpu_ctl_env &src_ctl = *internal::punned_cast<cpu_ctl_env*>(&src.my_cpu_ctl_env);
    new (&ctl) cpu_ctl_env( src_ctl );
    my_version_and_traits |= fp_settings;
}
#endif /* __TBB_FP_CONTEXT */

void task_group_context::register_pending_exception () {
    if ( my_cancellation_requested )
        return;
#if TBB_USE_EXCEPTIONS
    try {
        throw;
    } TbbCatchAll( this );
#endif /* TBB_USE_EXCEPTIONS */
}

#if __TBB_TASK_PRIORITY
void task_group_context::set_priority ( priority_t prio ) {
    __TBB_ASSERT( prio == priority_low || prio == priority_normal || prio == priority_high, "Invalid priority level value" );
    intptr_t p = normalize_priority(prio);
    if ( my_priority == p && !(my_state & task_group_context::may_have_children))
        return;
    my_priority = p;
    internal::generic_scheduler* s = governor::local_scheduler_if_initialized();
    if ( !s || !s->my_arena || !s->my_market->propagate_task_group_state(&task_group_context::my_priority, *this, p) )
        return;

    //! TODO: the arena of the calling thread might be unrelated;
    // need to find out the right arena for priority update.
    // The executing status check only guarantees being inside some working arena.
    if ( s->my_innermost_running_task->state() == task::executing )
        // Updating arena priority here does not eliminate necessity of checking each
        // task priority and updating arena priority if necessary before the task execution.
        // These checks will be necessary because:
        // a) set_priority() may be invoked before any tasks from this task group are spawned;
        // b) all spawned tasks from this task group are retrieved from the task pools.
        // These cases create a time window when arena priority may be lowered.
        s->my_market->update_arena_priority( *s->my_arena, p );
}

priority_t task_group_context::priority () const {
    return static_cast<priority_t>(priority_from_normalized_rep[my_priority]);
}
#endif /* __TBB_TASK_PRIORITY */

#endif /* __TBB_TASK_GROUP_CONTEXT */

} // namespace tbb
