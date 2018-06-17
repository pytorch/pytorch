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

// Do not include task.h directly. Use scheduler_common.h instead
#include "scheduler_common.h"
#include "governor.h"
#include "scheduler.h"
#include "itt_notify.h"

#include "tbb/cache_aligned_allocator.h"
#include "tbb/partitioner.h"

#include <new>

namespace tbb {

using namespace std;

namespace internal {

//------------------------------------------------------------------------
// Methods of allocate_root_proxy
//------------------------------------------------------------------------
task& allocate_root_proxy::allocate( size_t size ) {
    internal::generic_scheduler* v = governor::local_scheduler_weak();
    __TBB_ASSERT( v, "thread did not activate a task_scheduler_init object?" );
#if __TBB_TASK_GROUP_CONTEXT
    task_prefix& p = v->my_innermost_running_task->prefix();

    ITT_STACK_CREATE(p.context->itt_caller);
#endif
    // New root task becomes part of the currently running task's cancellation context
    return v->allocate_task( size, __TBB_CONTEXT_ARG(NULL, p.context) );
}

void allocate_root_proxy::free( task& task ) {
    internal::generic_scheduler* v = governor::local_scheduler_weak();
    __TBB_ASSERT( v, "thread does not have initialized task_scheduler_init object?" );
#if __TBB_TASK_GROUP_CONTEXT
    // No need to do anything here as long as there is no context -> task connection
#endif /* __TBB_TASK_GROUP_CONTEXT */
    v->free_task<local_task>( task );
}

#if __TBB_TASK_GROUP_CONTEXT
//------------------------------------------------------------------------
// Methods of allocate_root_with_context_proxy
//------------------------------------------------------------------------
task& allocate_root_with_context_proxy::allocate( size_t size ) const {
    internal::generic_scheduler* s = governor::local_scheduler_weak();
    __TBB_ASSERT( s, "Scheduler auto-initialization failed?" );
    __TBB_ASSERT( &my_context, "allocate_root(context) argument is a dereferenced NULL pointer" );
    task& t = s->allocate_task( size, NULL, &my_context );
    // Supported usage model prohibits concurrent initial binding. Thus we do not
    // need interlocked operations or fences to manipulate with my_context.my_kind
    if ( __TBB_load_relaxed(my_context.my_kind) == task_group_context::binding_required ) {
        // If we are in the outermost task dispatch loop of a master thread, then
        // there is nothing to bind this context to, and we skip the binding part
        // treating the context as isolated.
        if ( s->master_outermost_level() )
            __TBB_store_relaxed(my_context.my_kind, task_group_context::isolated);
        else
            my_context.bind_to( s );
    }
#if __TBB_FP_CONTEXT
    if ( __TBB_load_relaxed(my_context.my_kind) == task_group_context::isolated &&
            !(my_context.my_version_and_traits & task_group_context::fp_settings) )
        my_context.copy_fp_settings( *s->default_context() );
#endif
    ITT_STACK_CREATE(my_context.itt_caller);
    return t;
}

void allocate_root_with_context_proxy::free( task& task ) const {
    internal::generic_scheduler* v = governor::local_scheduler_weak();
    __TBB_ASSERT( v, "thread does not have initialized task_scheduler_init object?" );
    // No need to do anything here as long as unbinding is performed by context destructor only.
    v->free_task<local_task>( task );
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

//------------------------------------------------------------------------
// Methods of allocate_continuation_proxy
//------------------------------------------------------------------------
task& allocate_continuation_proxy::allocate( size_t size ) const {
    task* t = (task*)this;
    assert_task_valid(t);
    generic_scheduler* s = governor::local_scheduler_weak();
    task* parent = t->parent();
    t->prefix().parent = NULL;
    return s->allocate_task( size, __TBB_CONTEXT_ARG(parent, t->prefix().context) );
}

void allocate_continuation_proxy::free( task& mytask ) const {
    // Restore the parent as it was before the corresponding allocate was called.
    ((task*)this)->prefix().parent = mytask.parent();
    governor::local_scheduler_weak()->free_task<local_task>(mytask);
}

//------------------------------------------------------------------------
// Methods of allocate_child_proxy
//------------------------------------------------------------------------
task& allocate_child_proxy::allocate( size_t size ) const {
    task* t = (task*)this;
    assert_task_valid(t);
    generic_scheduler* s = governor::local_scheduler_weak();
    return s->allocate_task( size, __TBB_CONTEXT_ARG(t, t->prefix().context) );
}

void allocate_child_proxy::free( task& mytask ) const {
    governor::local_scheduler_weak()->free_task<local_task>(mytask);
}

//------------------------------------------------------------------------
// Methods of allocate_additional_child_of_proxy
//------------------------------------------------------------------------
task& allocate_additional_child_of_proxy::allocate( size_t size ) const {
    parent.increment_ref_count();
    generic_scheduler* s = governor::local_scheduler_weak();
    return s->allocate_task( size, __TBB_CONTEXT_ARG(&parent, parent.prefix().context) );
}

void allocate_additional_child_of_proxy::free( task& task ) const {
    // Undo the increment.  We do not check the result of the fetch-and-decrement.
    // We could consider be spawning the task if the fetch-and-decrement returns 1.
    // But we do not know that was the programmer's intention.
    // Furthermore, if it was the programmer's intention, the program has a fundamental
    // race condition (that we warn about in Reference manual), because the
    // reference count might have become zero before the corresponding call to
    // allocate_additional_child_of_proxy::allocate.
    parent.internal_decrement_ref_count();
    governor::local_scheduler_weak()->free_task<local_task>(task);
}

//------------------------------------------------------------------------
// Support for auto_partitioner
//------------------------------------------------------------------------
size_t get_initial_auto_partitioner_divisor() {
    const size_t X_FACTOR = 4;
    return X_FACTOR * governor::local_scheduler()->max_threads_in_arena();
}

//------------------------------------------------------------------------
// Methods of affinity_partitioner_base_v3
//------------------------------------------------------------------------
void affinity_partitioner_base_v3::resize( unsigned factor ) {
    // Check factor to avoid asking for number of workers while there might be no arena.
    size_t new_size = factor ? factor*governor::local_scheduler()->max_threads_in_arena() : 0;
    if( new_size!=my_size ) {
        if( my_array ) {
            NFS_Free( my_array );
            // Following two assignments must be done here for sake of exception safety.
            my_array = NULL;
            my_size = 0;
        }
        if( new_size ) {
            my_array = static_cast<affinity_id*>(NFS_Allocate(new_size,sizeof(affinity_id), NULL ));
            memset( my_array, 0, sizeof(affinity_id)*new_size );
            my_size = new_size;
        }
    }
}

} // namespace internal

using namespace tbb::internal;

//------------------------------------------------------------------------
// task
//------------------------------------------------------------------------

void task::internal_set_ref_count( int count ) {
    __TBB_ASSERT( count>=0, "count must not be negative" );
    task_prefix &p = prefix();
    __TBB_ASSERT(p.ref_count==1 && p.state==allocated && self().parent()==this
        || !(p.extra_state & es_ref_count_active), "ref_count race detected");
    ITT_NOTIFY(sync_releasing, &p.ref_count);
    p.ref_count = count;
}

internal::reference_count task::internal_decrement_ref_count() {
    ITT_NOTIFY( sync_releasing, &prefix().ref_count );
    internal::reference_count k = __TBB_FetchAndDecrementWrelease( &prefix().ref_count );
    __TBB_ASSERT( k>=1, "task's reference count underflowed" );
    if( k==1 )
        ITT_NOTIFY( sync_acquired, &prefix().ref_count );
    return k-1;
}

task& task::self() {
    generic_scheduler *v = governor::local_scheduler_weak();
    v->assert_task_pool_valid();
    __TBB_ASSERT( v->my_innermost_running_task, NULL );
    return *v->my_innermost_running_task;
}

bool task::is_owned_by_current_thread() const {
    return true;
}

void interface5::internal::task_base::destroy( task& victim ) {
    // 1 may be a guard reference for wait_for_all, which was not reset because
    // of concurrent_wait mode or because prepared root task was not actually used
    // for spawning tasks (as in structured_task_group).
    __TBB_ASSERT( (intptr_t)victim.prefix().ref_count <= 1, "Task being destroyed must not have children" );
    __TBB_ASSERT( victim.state()==task::allocated, "illegal state for victim task" );
    task* parent = victim.parent();
    victim.~task();
    if( parent ) {
        __TBB_ASSERT( parent->state()!=task::freed && parent->state()!=task::ready,
                      "attempt to destroy child of running or corrupted parent?" );
        // 'reexecute' and 'executing' are also signs of a race condition, since most tasks
        // set their ref_count upon entry but "es_ref_count_active" should detect this
        parent->internal_decrement_ref_count();
        // Even if the last reference to *parent is removed, it should not be spawned (documented behavior).
    }
    governor::local_scheduler_weak()->free_task<no_cache>( victim );
}

void task::spawn_and_wait_for_all( task_list& list ) {
    generic_scheduler* s = governor::local_scheduler();
    task* t = list.first;
    if( t ) {
        if( &t->prefix().next!=list.next_ptr )
            s->local_spawn( t->prefix().next, *list.next_ptr );
        list.clear();
    }
    s->local_wait_for_all( *this, t );
}

/** Defined out of line so that compiler does not replicate task's vtable.
    It's pointless to define it inline anyway, because all call sites to it are virtual calls
    that the compiler is unlikely to optimize. */
void task::note_affinity( affinity_id ) {
}

#if __TBB_TASK_GROUP_CONTEXT
void task::change_group ( task_group_context& ctx ) {
    prefix().context = &ctx;
    internal::generic_scheduler* s = governor::local_scheduler_weak();
    if ( __TBB_load_relaxed(ctx.my_kind) == task_group_context::binding_required ) {
        // If we are in the outermost task dispatch loop of a master thread, then
        // there is nothing to bind this context to, and we skip the binding part
        // treating the context as isolated.
        if ( s->master_outermost_level() )
            __TBB_store_relaxed(ctx.my_kind, task_group_context::isolated);
        else
            ctx.bind_to( s );
    }
#if __TBB_FP_CONTEXT
    if ( __TBB_load_relaxed(ctx.my_kind) == task_group_context::isolated &&
            !(ctx.my_version_and_traits & task_group_context::fp_settings) )
        ctx.copy_fp_settings( *s->default_context() );
#endif
    ITT_STACK_CREATE(ctx.itt_caller);
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

} // namespace tbb

