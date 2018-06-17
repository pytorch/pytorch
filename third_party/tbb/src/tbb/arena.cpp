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

#include "tbb/global_control.h" // thread_stack_size

#include "scheduler.h"
#include "governor.h"
#include "arena.h"
#include "itt_notify.h"
#include "semaphore.h"
#include "tbb/internal/_flow_graph_impl.h"

#include <functional>

#if __TBB_STATISTICS_STDOUT
#include <cstdio>
#endif

namespace tbb {
namespace internal {

// put it here in order to enable compiler to inline it into arena::process and nested_arena_entry
void generic_scheduler::attach_arena( arena* a, size_t index, bool is_master ) {
    __TBB_ASSERT( a->my_market == my_market, NULL );
    my_arena = a;
    my_arena_index = index;
    my_arena_slot = a->my_slots + index;
    attach_mailbox( affinity_id(index+1) );
    if ( is_master && my_inbox.is_idle_state( true ) ) {
        // Master enters an arena with its own task to be executed. It means that master is not
        // going to enter stealing loop and take affinity tasks.
        my_inbox.set_is_idle( false );
    }
#if __TBB_TASK_GROUP_CONTEXT
    // Context to be used by root tasks by default (if the user has not specified one).
    if( !is_master )
        my_dummy_task->prefix().context = a->my_default_ctx;
#endif /* __TBB_TASK_GROUP_CONTEXT */
#if __TBB_TASK_PRIORITY
    // In the current implementation master threads continue processing even when
    // there are other masters with higher priority. Only TBB worker threads are
    // redistributed between arenas based on the latters' priority. Thus master
    // threads use arena's top priority as a reference point (in contrast to workers
    // that use my_market->my_global_top_priority).
    if( is_master ) {
        my_ref_top_priority = &a->my_top_priority;
        my_ref_reload_epoch = &a->my_reload_epoch;
    }
    my_local_reload_epoch = *my_ref_reload_epoch;
    __TBB_ASSERT( !my_offloaded_tasks, NULL );
#endif /* __TBB_TASK_PRIORITY */
}

inline static bool occupy_slot( generic_scheduler*& slot, generic_scheduler& s ) {
    return !slot && as_atomic( slot ).compare_and_swap( &s, NULL ) == NULL;
}

size_t arena::occupy_free_slot_in_range( generic_scheduler& s, size_t lower, size_t upper ) {
    if ( lower >= upper ) return out_of_arena;
    // Start search for an empty slot from the one we occupied the last time
    size_t index = s.my_arena_index;
    if ( index < lower || index >= upper ) index = s.my_random.get() % (upper - lower) + lower;
    __TBB_ASSERT( index >= lower && index < upper, NULL );
    // Find a free slot
    for ( size_t i = index; i < upper; ++i )
        if ( occupy_slot(my_slots[i].my_scheduler, s) ) return i;
    for ( size_t i = lower; i < index; ++i )
        if ( occupy_slot(my_slots[i].my_scheduler, s) ) return i;
    return out_of_arena;
}

template <bool as_worker>
size_t arena::occupy_free_slot( generic_scheduler& s ) {
    // Firstly, masters try to occupy reserved slots
    size_t index = as_worker ? out_of_arena : occupy_free_slot_in_range( s, 0, my_num_reserved_slots );
    if ( index == out_of_arena ) {
        // Secondly, all threads try to occupy all non-reserved slots
        index = occupy_free_slot_in_range( s, my_num_reserved_slots, my_num_slots );
        // Likely this arena is already saturated
        if ( index == out_of_arena )
            return out_of_arena;
    }

    ITT_NOTIFY(sync_acquired, my_slots + index);
    atomic_update( my_limit, (unsigned)(index + 1), std::less<unsigned>() );
    return index;
}

void arena::process( generic_scheduler& s ) {
    __TBB_ASSERT( is_alive(my_guard), NULL );
    __TBB_ASSERT( governor::is_set(&s), NULL );
    __TBB_ASSERT( s.my_innermost_running_task == s.my_dummy_task, NULL );
    __TBB_ASSERT( s.worker_outermost_level(), NULL );

    __TBB_ASSERT( my_num_slots > 1, NULL );

    size_t index = occupy_free_slot</*as_worker*/true>( s );
    if ( index == out_of_arena )
        goto quit;

    __TBB_ASSERT( index >= my_num_reserved_slots, "Workers cannot occupy reserved slots" );
    s.attach_arena( this, index, /*is_master*/false );

#if !__TBB_FP_CONTEXT
    my_cpu_ctl_env.set_env();
#endif

#if __TBB_ARENA_OBSERVER
    __TBB_ASSERT( !s.my_last_local_observer, "There cannot be notified local observers when entering arena" );
    my_observers.notify_entry_observers( s.my_last_local_observer, /*worker=*/true );
#endif /* __TBB_ARENA_OBSERVER */

    // Task pool can be marked as non-empty if the worker occupies the slot left by a master.
    if ( s.my_arena_slot->task_pool != EmptyTaskPool ) {
        __TBB_ASSERT( s.my_inbox.is_idle_state(false), NULL );
        s.local_wait_for_all( *s.my_dummy_task, NULL );
        __TBB_ASSERT( s.my_inbox.is_idle_state(true), NULL );
    }

    for ( ;; ) {
        __TBB_ASSERT( s.my_innermost_running_task == s.my_dummy_task, NULL );
        __TBB_ASSERT( s.worker_outermost_level(), NULL );
        __TBB_ASSERT( is_alive(my_guard), NULL );
        __TBB_ASSERT( s.is_quiescent_local_task_pool_reset(),
                      "Worker cannot leave arena while its task pool is not reset" );
        __TBB_ASSERT( s.my_arena_slot->task_pool == EmptyTaskPool, "Empty task pool is not marked appropriately" );
        // This check prevents relinquishing more than necessary workers because
        // of the non-atomicity of the decision making procedure
        if ( num_workers_active() > my_num_workers_allotted
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
             || recall_by_mandatory_request()
#endif
            )
            break;
        // Try to steal a task.
        // Passing reference count is technically unnecessary in this context,
        // but omitting it here would add checks inside the function.
        task* t = s.receive_or_steal_task( __TBB_ISOLATION_ARG( s.my_dummy_task->prefix().ref_count, no_isolation ) );
        if (t) {
            // A side effect of receive_or_steal_task is that my_innermost_running_task can be set.
            // But for the outermost dispatch loop it has to be a dummy task.
            s.my_innermost_running_task = s.my_dummy_task;
            s.local_wait_for_all(*s.my_dummy_task,t);
        }
    }
#if __TBB_ARENA_OBSERVER
    my_observers.notify_exit_observers( s.my_last_local_observer, /*worker=*/true );
    s.my_last_local_observer = NULL;
#endif /* __TBB_ARENA_OBSERVER */
#if __TBB_TASK_PRIORITY
    if ( s.my_offloaded_tasks )
        orphan_offloaded_tasks( s );
#endif /* __TBB_TASK_PRIORITY */
#if __TBB_STATISTICS
    ++s.my_counters.arena_roundtrips;
    *my_slots[index].my_counters += s.my_counters;
    s.my_counters.reset();
#endif /* __TBB_STATISTICS */
    __TBB_store_with_release( my_slots[index].my_scheduler, (generic_scheduler*)NULL );
    s.my_arena_slot = 0; // detached from slot
    s.my_inbox.detach();
    __TBB_ASSERT( s.my_inbox.is_idle_state(true), NULL );
    __TBB_ASSERT( s.my_innermost_running_task == s.my_dummy_task, NULL );
    __TBB_ASSERT( s.worker_outermost_level(), NULL );
    __TBB_ASSERT( is_alive(my_guard), NULL );
quit:
    // In contrast to earlier versions of TBB (before 3.0 U5) now it is possible
    // that arena may be temporarily left unpopulated by threads. See comments in
    // arena::on_thread_leaving() for more details.
    on_thread_leaving<ref_worker>();
}

arena::arena ( market& m, unsigned num_slots, unsigned num_reserved_slots ) {
    __TBB_ASSERT( !my_guard, "improperly allocated arena?" );
    __TBB_ASSERT( sizeof(my_slots[0]) % NFS_GetLineSize()==0, "arena::slot size not multiple of cache line size" );
    __TBB_ASSERT( (uintptr_t)this % NFS_GetLineSize()==0, "arena misaligned" );
#if __TBB_TASK_PRIORITY
    __TBB_ASSERT( !my_reload_epoch && !my_orphaned_tasks && !my_skipped_fifo_priority, "New arena object is not zeroed" );
#endif /* __TBB_TASK_PRIORITY */
    my_market = &m;
    my_limit = 1;
    // Two slots are mandatory: for the master, and for 1 worker (required to support starvation resistant tasks).
    my_num_slots = num_arena_slots(num_slots);
    my_num_reserved_slots = num_reserved_slots;
    my_max_num_workers = num_slots-num_reserved_slots;
    my_references = ref_external; // accounts for the master
#if __TBB_TASK_PRIORITY
    my_bottom_priority = my_top_priority = normalized_normal_priority;
#endif /* __TBB_TASK_PRIORITY */
    my_aba_epoch = m.my_arenas_aba_epoch;
#if __TBB_ARENA_OBSERVER
    my_observers.my_arena = this;
#endif
    __TBB_ASSERT ( my_max_num_workers <= my_num_slots, NULL );
    // Construct slots. Mark internal synchronization elements for the tools.
    for( unsigned i = 0; i < my_num_slots; ++i ) {
        __TBB_ASSERT( !my_slots[i].my_scheduler && !my_slots[i].task_pool, NULL );
        __TBB_ASSERT( !my_slots[i].task_pool_ptr, NULL );
        __TBB_ASSERT( !my_slots[i].my_task_pool_size, NULL );
        ITT_SYNC_CREATE(my_slots + i, SyncType_Scheduler, SyncObj_WorkerTaskPool);
        mailbox(i+1).construct();
        ITT_SYNC_CREATE(&mailbox(i+1), SyncType_Scheduler, SyncObj_Mailbox);
        my_slots[i].hint_for_pop = i;
#if __TBB_STATISTICS
        my_slots[i].my_counters = new ( NFS_Allocate(1, sizeof(statistics_counters), NULL) ) statistics_counters;
#endif /* __TBB_STATISTICS */
    }
    my_task_stream.initialize(my_num_slots);
    ITT_SYNC_CREATE(&my_task_stream, SyncType_Scheduler, SyncObj_TaskStream);
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    my_concurrency_mode = cm_normal;
#endif
#if !__TBB_FP_CONTEXT
    my_cpu_ctl_env.get_env();
#endif
}

arena& arena::allocate_arena( market& m, unsigned num_slots, unsigned num_reserved_slots ) {
    __TBB_ASSERT( sizeof(base_type) + sizeof(arena_slot) == sizeof(arena), "All arena data fields must go to arena_base" );
    __TBB_ASSERT( sizeof(base_type) % NFS_GetLineSize() == 0, "arena slots area misaligned: wrong padding" );
    __TBB_ASSERT( sizeof(mail_outbox) == NFS_MaxLineSize, "Mailbox padding is wrong" );
    size_t n = allocation_size(num_arena_slots(num_slots));
    unsigned char* storage = (unsigned char*)NFS_Allocate( 1, n, NULL );
    // Zero all slots to indicate that they are empty
    memset( storage, 0, n );
    return *new( storage + num_arena_slots(num_slots) * sizeof(mail_outbox) ) arena(m, num_slots, num_reserved_slots);
}

void arena::free_arena () {
    __TBB_ASSERT( is_alive(my_guard), NULL );
    __TBB_ASSERT( !my_references, "There are threads in the dying arena" );
    __TBB_ASSERT( !my_num_workers_requested && !my_num_workers_allotted, "Dying arena requests workers" );
    __TBB_ASSERT( my_pool_state == SNAPSHOT_EMPTY || !my_max_num_workers, "Inconsistent state of a dying arena" );
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    __TBB_ASSERT( my_concurrency_mode != cm_enforced_global, NULL );
#endif
#if !__TBB_STATISTICS_EARLY_DUMP
    GATHER_STATISTIC( dump_arena_statistics() );
#endif
    poison_value( my_guard );
    intptr_t drained = 0;
    for ( unsigned i = 0; i < my_num_slots; ++i ) {
        __TBB_ASSERT( !my_slots[i].my_scheduler, "arena slot is not empty" );
        // TODO: understand the assertion and modify
        // __TBB_ASSERT( my_slots[i].task_pool == EmptyTaskPool, NULL );
        __TBB_ASSERT( my_slots[i].head == my_slots[i].tail, NULL ); // TODO: replace by is_quiescent_local_task_pool_empty
        my_slots[i].free_task_pool();
#if __TBB_STATISTICS
        NFS_Free( my_slots[i].my_counters );
#endif /* __TBB_STATISTICS */
        drained += mailbox(i+1).drain();
    }
    __TBB_ASSERT( my_task_stream.drain()==0, "Not all enqueued tasks were executed");
#if __TBB_COUNT_TASK_NODES
    my_market->update_task_node_count( -drained );
#endif /* __TBB_COUNT_TASK_NODES */
    // remove an internal reference
    my_market->release( /*is_public=*/false, /*blocking_terminate=*/false );
#if __TBB_TASK_GROUP_CONTEXT
    __TBB_ASSERT( my_default_ctx, "Master thread never entered the arena?" );
    my_default_ctx->~task_group_context();
    NFS_Free(my_default_ctx);
#endif /* __TBB_TASK_GROUP_CONTEXT */
#if __TBB_ARENA_OBSERVER
    if ( !my_observers.empty() )
        my_observers.clear();
#endif /* __TBB_ARENA_OBSERVER */
    void* storage  = &mailbox(my_num_slots);
    __TBB_ASSERT( my_references == 0, NULL );
    __TBB_ASSERT( my_pool_state == SNAPSHOT_EMPTY || !my_max_num_workers, NULL );
    this->~arena();
#if TBB_USE_ASSERT > 1
    memset( storage, 0, allocation_size(my_num_slots) );
#endif /* TBB_USE_ASSERT */
    NFS_Free( storage );
}

#if __TBB_STATISTICS
void arena::dump_arena_statistics () {
    statistics_counters total;
    for( unsigned i = 0; i < my_num_slots; ++i ) {
#if __TBB_STATISTICS_EARLY_DUMP
        generic_scheduler* s = my_slots[i].my_scheduler;
        if ( s )
            *my_slots[i].my_counters += s->my_counters;
#else
        __TBB_ASSERT( !my_slots[i].my_scheduler, NULL );
#endif
        if ( i != 0 ) {
            total += *my_slots[i].my_counters;
            dump_statistics( *my_slots[i].my_counters, i );
        }
    }
    dump_statistics( *my_slots[0].my_counters, 0 );
#if __TBB_STATISTICS_STDOUT
#if !__TBB_STATISTICS_TOTALS_ONLY
    printf( "----------------------------------------------\n" );
#endif
    dump_statistics( total, workers_counters_total );
    total += *my_slots[0].my_counters;
    dump_statistics( total, arena_counters_total );
#if !__TBB_STATISTICS_TOTALS_ONLY
    printf( "==============================================\n" );
#endif
#endif /* __TBB_STATISTICS_STDOUT */
}
#endif /* __TBB_STATISTICS */

#if __TBB_TASK_PRIORITY
// The method inspects a scheduler to determine:
// 1. if it has tasks that can be retrieved and executed (via the return value);
// 2. if it has any tasks at all, including those of lower priority (via tasks_present);
// 3. if it is able to work with enqueued tasks (via dequeuing_possible).
inline bool arena::may_have_tasks ( generic_scheduler* s, bool& tasks_present, bool& dequeuing_possible ) {
    if ( !s || s->my_arena != this )
        return false;
    dequeuing_possible |= s->worker_outermost_level();
    if ( s->my_pool_reshuffling_pending ) {
        // This primary task pool is nonempty and may contain tasks at the current
        // priority level. Its owner is winnowing lower priority tasks at the moment.
        tasks_present = true;
        return true;
    }
    if ( s->my_offloaded_tasks ) {
        tasks_present = true;
        if ( s->my_local_reload_epoch < *s->my_ref_reload_epoch ) {
            // This scheduler's offload area is nonempty and may contain tasks at the
            // current priority level.
            return true;
        }
    }
    return false;
}

void arena::orphan_offloaded_tasks(generic_scheduler& s) {
    __TBB_ASSERT( s.my_offloaded_tasks, NULL );
    GATHER_STATISTIC( ++s.my_counters.prio_orphanings );
    ++my_abandonment_epoch;
    __TBB_ASSERT( s.my_offloaded_task_list_tail_link && !*s.my_offloaded_task_list_tail_link, NULL );
    task* orphans;
    do {
        orphans = const_cast<task*>(my_orphaned_tasks);
        *s.my_offloaded_task_list_tail_link = orphans;
    } while ( as_atomic(my_orphaned_tasks).compare_and_swap(s.my_offloaded_tasks, orphans) != orphans );
    s.my_offloaded_tasks = NULL;
#if TBB_USE_ASSERT
    s.my_offloaded_task_list_tail_link = NULL;
#endif /* TBB_USE_ASSERT */
}
#endif /* __TBB_TASK_PRIORITY */

bool arena::has_enqueued_tasks() {
    // Look for enqueued tasks at all priority levels
    for ( int p = 0; p < num_priority_levels; ++p )
        if ( !my_task_stream.empty(p) )
            return true;
    return false;
}

void arena::restore_priority_if_need() {
    // Check for the presence of enqueued tasks "lost" on some of
    // priority levels because updating arena priority and switching
    // arena into "populated" (FULL) state happen non-atomically.
    // Imposing atomicity would require task::enqueue() to use a lock,
    // which is unacceptable.
    if ( has_enqueued_tasks() ) {
        advertise_new_work<work_enqueued>();
#if __TBB_TASK_PRIORITY
        // update_arena_priority() expects non-zero arena::my_num_workers_requested,
        // so must be called after advertise_new_work<work_enqueued>()
        for ( int p = 0; p < num_priority_levels; ++p )
            if ( !my_task_stream.empty(p) ) {
                if ( p < my_bottom_priority || p > my_top_priority )
                    my_market->update_arena_priority(*this, p);
            }
#endif
    }
}

bool arena::is_out_of_work() {
    // TODO: rework it to return at least a hint about where a task was found; better if the task itself.
    for(;;) {
        pool_state_t snapshot = my_pool_state;
        switch( snapshot ) {
            case SNAPSHOT_EMPTY:
                return true;
            case SNAPSHOT_FULL: {
                // Use unique id for "busy" in order to avoid ABA problems.
                const pool_state_t busy = pool_state_t(&busy);
                // Request permission to take snapshot
                if( my_pool_state.compare_and_swap( busy, SNAPSHOT_FULL )==SNAPSHOT_FULL ) {
                    // Got permission. Take the snapshot.
                    // NOTE: This is not a lock, as the state can be set to FULL at
                    //       any moment by a thread that spawns/enqueues new task.
                    size_t n = my_limit;
                    // Make local copies of volatile parameters. Their change during
                    // snapshot taking procedure invalidates the attempt, and returns
                    // this thread into the dispatch loop.
#if __TBB_TASK_PRIORITY
                    uintptr_t reload_epoch = __TBB_load_with_acquire( my_reload_epoch );
                    intptr_t top_priority = my_top_priority;
                    // Inspect primary task pools first
#endif /* __TBB_TASK_PRIORITY */
                    size_t k;
                    for( k=0; k<n; ++k ) {
                        if( my_slots[k].task_pool != EmptyTaskPool &&
                            __TBB_load_relaxed(my_slots[k].head) < __TBB_load_relaxed(my_slots[k].tail) )
                        {
                            // k-th primary task pool is nonempty and does contain tasks.
                            break;
                        }
                        if( my_pool_state!=busy )
                            return false; // the work was published
                    }
                    __TBB_ASSERT( k <= n, NULL );
                    bool work_absent = k == n;
#if __TBB_TASK_PRIORITY
                    // Variable tasks_present indicates presence of tasks at any priority
                    // level, while work_absent refers only to the current priority.
                    bool tasks_present = !work_absent || my_orphaned_tasks;
                    bool dequeuing_possible = false;
                    if ( work_absent ) {
                        // Check for the possibility that recent priority changes
                        // brought some tasks to the current priority level

                        uintptr_t abandonment_epoch = my_abandonment_epoch;
                        // Master thread's scheduler needs special handling as it
                        // may be destroyed at any moment (workers' schedulers are
                        // guaranteed to be alive while at least one thread is in arena).
                        // The lock below excludes concurrency with task group state change
                        // propagation and guarantees lifetime of the master thread.
                        the_context_state_propagation_mutex.lock();
                        work_absent = !may_have_tasks( my_slots[0].my_scheduler, tasks_present, dequeuing_possible );
                        the_context_state_propagation_mutex.unlock();
                        // The following loop is subject to data races. While k-th slot's
                        // scheduler is being examined, corresponding worker can either
                        // leave to RML or migrate to another arena.
                        // But the races are not prevented because all of them are benign.
                        // First, the code relies on the fact that worker thread's scheduler
                        // object persists until the whole library is deinitialized.
                        // Second, in the worst case the races can only cause another
                        // round of stealing attempts to be undertaken. Introducing complex
                        // synchronization into this coldest part of the scheduler's control
                        // flow does not seem to make sense because it both is unlikely to
                        // ever have any observable performance effect, and will require
                        // additional synchronization code on the hotter paths.
                        for( k = 1; work_absent && k < n; ++k ) {
                            if( my_pool_state!=busy )
                                return false; // the work was published
                            work_absent = !may_have_tasks( my_slots[k].my_scheduler, tasks_present, dequeuing_possible );
                        }
                        // Preclude premature switching arena off because of a race in the previous loop.
                        work_absent = work_absent
                                      && !__TBB_load_with_acquire(my_orphaned_tasks)
                                      && abandonment_epoch == my_abandonment_epoch;
                    }
#endif /* __TBB_TASK_PRIORITY */
                    // Test and test-and-set.
                    if( my_pool_state==busy ) {
#if __TBB_TASK_PRIORITY
                        bool no_fifo_tasks = my_task_stream.empty(top_priority);
                        work_absent = work_absent && (!dequeuing_possible || no_fifo_tasks)
                                      && top_priority == my_top_priority && reload_epoch == my_reload_epoch;
#else
                        bool no_fifo_tasks = my_task_stream.empty(0);
                        work_absent = work_absent && no_fifo_tasks;
#endif /* __TBB_TASK_PRIORITY */
                        if( work_absent ) {
#if __TBB_TASK_PRIORITY
                            if ( top_priority > my_bottom_priority ) {
                                if ( my_market->lower_arena_priority(*this, top_priority - 1, reload_epoch)
                                     && !my_task_stream.empty(top_priority) )
                                {
                                    atomic_update( my_skipped_fifo_priority, top_priority, std::less<intptr_t>());
                                }
                            }
                            else if ( !tasks_present && !my_orphaned_tasks && no_fifo_tasks ) {
#endif /* __TBB_TASK_PRIORITY */
                                // save current demand value before setting SNAPSHOT_EMPTY,
                                // to avoid race with advertise_new_work.
                                int current_demand = (int)my_max_num_workers;
                                if( my_pool_state.compare_and_swap( SNAPSHOT_EMPTY, busy )==busy ) {
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
                                    if( my_concurrency_mode==cm_enforced_global  ) {
                                        // adjust_demand() called inside, if needed
                                        my_market->mandatory_concurrency_disable( this );
                                    } else
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */
                                    {
                                        // This thread transitioned pool to empty state, and thus is
                                        // responsible for telling the market that there is no work to do.
                                        my_market->adjust_demand( *this, -current_demand );
                                    }
                                    restore_priority_if_need();
                                    return true;
                                }
                                return false;
#if __TBB_TASK_PRIORITY
                            }
#endif /* __TBB_TASK_PRIORITY */
                        }
                        // Undo previous transition SNAPSHOT_FULL-->busy, unless another thread undid it.
                        my_pool_state.compare_and_swap( SNAPSHOT_FULL, busy );
                    }
                }
                return false;
            }
            default:
                // Another thread is taking a snapshot.
                return false;
        }
    }
}

#if __TBB_COUNT_TASK_NODES
intptr_t arena::workers_task_node_count() {
    intptr_t result = 0;
    for( unsigned i = 1; i < my_num_slots; ++i ) {
        generic_scheduler* s = my_slots[i].my_scheduler;
        if( s )
            result += s->my_task_node_count;
    }
    return result;
}
#endif /* __TBB_COUNT_TASK_NODES */

void arena::enqueue_task( task& t, intptr_t prio, FastRandom &random )
{
#if __TBB_RECYCLE_TO_ENQUEUE
    __TBB_ASSERT( t.state()==task::allocated || t.state()==task::to_enqueue, "attempt to enqueue task with inappropriate state" );
#else
    __TBB_ASSERT( t.state()==task::allocated, "attempt to enqueue task that is not in 'allocated' state" );
#endif
    t.prefix().state = task::ready;
    t.prefix().extra_state |= es_task_enqueued; // enqueued task marker

#if TBB_USE_ASSERT
    if( task* parent = t.parent() ) {
        internal::reference_count ref_count = parent->prefix().ref_count;
        __TBB_ASSERT( ref_count!=0, "attempt to enqueue task whose parent has a ref_count==0 (forgot to set_ref_count?)" );
        __TBB_ASSERT( ref_count>0, "attempt to enqueue task whose parent has a ref_count<0" );
        parent->prefix().extra_state |= es_ref_count_active;
    }
    __TBB_ASSERT(t.prefix().affinity==affinity_id(0), "affinity is ignored for enqueued tasks");
#endif /* TBB_USE_ASSERT */

    ITT_NOTIFY(sync_releasing, &my_task_stream);
#if __TBB_TASK_PRIORITY
    intptr_t p = prio ? normalize_priority(priority_t(prio)) : normalized_normal_priority;
    assert_priority_valid(p);
    my_task_stream.push( &t, p, random );
    if ( p != my_top_priority )
        my_market->update_arena_priority( *this, p );
#else /* !__TBB_TASK_PRIORITY */
    __TBB_ASSERT_EX(prio == 0, "the library is not configured to respect the task priority");
    my_task_stream.push( &t, 0, random );
#endif /* !__TBB_TASK_PRIORITY */
    advertise_new_work<work_enqueued>();
#if __TBB_TASK_PRIORITY
    if ( p != my_top_priority )
        my_market->update_arena_priority( *this, p );
#endif /* __TBB_TASK_PRIORITY */
}

class nested_arena_context : no_copy {
public:
    nested_arena_context(generic_scheduler *s, arena* a, size_t slot_index, bool type, bool same)
        : my_scheduler(*s), my_orig_ctx(NULL), same_arena(same) {
        if (same_arena) {
            my_orig_state.my_properties = my_scheduler.my_properties;
            my_orig_state.my_innermost_running_task = my_scheduler.my_innermost_running_task;
            mimic_outermost_level(a, type);
        } else {
            my_orig_state = *s;
            mimic_outermost_level(a, type);
            s->nested_arena_entry(a, slot_index);
        }
    }
    ~nested_arena_context() {
#if __TBB_TASK_GROUP_CONTEXT
        my_scheduler.my_dummy_task->prefix().context = my_orig_ctx; // restore context of dummy task
#endif
        if (same_arena) {
            my_scheduler.my_properties = my_orig_state.my_properties;
            my_scheduler.my_innermost_running_task = my_orig_state.my_innermost_running_task;
        } else {
            my_scheduler.nested_arena_exit();
            static_cast<scheduler_state&>(my_scheduler) = my_orig_state; // restore arena settings 
#if __TBB_TASK_PRIORITY
            my_scheduler.my_local_reload_epoch = *my_orig_state.my_ref_reload_epoch;
#endif
            governor::assume_scheduler(&my_scheduler);
        }
    }

private:
    generic_scheduler &my_scheduler;
    scheduler_state my_orig_state;
    task_group_context *my_orig_ctx;
    const bool same_arena;

    void mimic_outermost_level(arena* a, bool type) {
        my_scheduler.my_properties.outermost = true;
        my_scheduler.my_properties.type = type;
        my_scheduler.my_innermost_running_task = my_scheduler.my_dummy_task;
#if __TBB_TASK_GROUP_CONTEXT
        // Save dummy's context and replace it by arena's context
        my_orig_ctx = my_scheduler.my_dummy_task->prefix().context;
        my_scheduler.my_dummy_task->prefix().context = a->my_default_ctx;
#endif
    }
};

void generic_scheduler::nested_arena_entry(arena* a, size_t slot_index) {
    __TBB_ASSERT( is_alive(a->my_guard), NULL );
    __TBB_ASSERT( a!=my_arena, NULL);

    // overwrite arena settings
#if __TBB_TASK_PRIORITY
    if ( my_offloaded_tasks )
        my_arena->orphan_offloaded_tasks( *this );
    my_offloaded_tasks = NULL;
#endif /* __TBB_TASK_PRIORITY */
    attach_arena( a, slot_index, /*is_master*/true );
    __TBB_ASSERT( my_arena == a, NULL );
    governor::assume_scheduler( this );
    // TODO? ITT_NOTIFY(sync_acquired, a->my_slots + index);
    // TODO: it requires market to have P workers (not P-1)
    // TODO: a preempted worker should be excluded from assignment to other arenas e.g. my_slack--
    if( !is_worker() && slot_index >= my_arena->my_num_reserved_slots )
        my_arena->my_market->adjust_demand(*my_arena, -1);
#if __TBB_ARENA_OBSERVER
    my_last_local_observer = 0; // TODO: try optimize number of calls
    my_arena->my_observers.notify_entry_observers( my_last_local_observer, /*worker=*/false );
#endif
}

void generic_scheduler::nested_arena_exit() {
#if __TBB_ARENA_OBSERVER
    my_arena->my_observers.notify_exit_observers( my_last_local_observer, /*worker=*/false );
#endif /* __TBB_ARENA_OBSERVER */
#if __TBB_TASK_PRIORITY
    if ( my_offloaded_tasks )
        my_arena->orphan_offloaded_tasks( *this );
#endif
    if( !is_worker() && my_arena_index >= my_arena->my_num_reserved_slots )
        my_arena->my_market->adjust_demand(*my_arena, 1);
    // Free the master slot.
    __TBB_ASSERT(my_arena->my_slots[my_arena_index].my_scheduler, "A slot is already empty");
    __TBB_store_with_release(my_arena->my_slots[my_arena_index].my_scheduler, (generic_scheduler*)NULL);
    my_arena->my_exit_monitors.notify_one(); // do not relax!
}

void generic_scheduler::wait_until_empty() {
    my_dummy_task->prefix().ref_count++; // prevents exit from local_wait_for_all when local work is done enforcing the stealing
    while( my_arena->my_pool_state != arena::SNAPSHOT_EMPTY )
        local_wait_for_all(*my_dummy_task, NULL);
    my_dummy_task->prefix().ref_count--;
}

} // namespace internal
} // namespace tbb

#include "scheduler_utility.h"
#include "tbb/task_arena.h" // task_arena_base

namespace tbb {
namespace interface7 {
namespace internal {

void task_arena_base::internal_initialize( ) {
    governor::one_time_init();
    if( my_max_concurrency < 1 )
        my_max_concurrency = (int)governor::default_num_threads();
    __TBB_ASSERT( my_master_slots <= (unsigned)my_max_concurrency, "Number of slots reserved for master should not exceed arena concurrency");
    arena* new_arena = market::create_arena( my_max_concurrency, my_master_slots, 0 );
    // add an internal market reference; a public reference was added in create_arena
    market &m = market::global_market( /*is_public=*/false );
    // allocate default context for task_arena
#if __TBB_TASK_GROUP_CONTEXT
    new_arena->my_default_ctx = new ( NFS_Allocate(1, sizeof(task_group_context), NULL) )
            task_group_context( task_group_context::isolated, task_group_context::default_traits );
#if __TBB_FP_CONTEXT
    new_arena->my_default_ctx->capture_fp_settings();
#endif
#endif /* __TBB_TASK_GROUP_CONTEXT */
    // threads might race to initialize the arena
    if(as_atomic(my_arena).compare_and_swap(new_arena, NULL) != NULL) {
        __TBB_ASSERT(my_arena, NULL); // another thread won the race
        // release public market reference
        m.release( /*is_public=*/true, /*blocking_terminate=*/false );
        new_arena->on_thread_leaving<arena::ref_external>(); // destroy unneeded arena
#if __TBB_TASK_GROUP_CONTEXT
        spin_wait_while_eq(my_context, (task_group_context*)NULL);
    } else {
        new_arena->my_default_ctx->my_version_and_traits |= my_version_and_traits & exact_exception_flag;
        as_atomic(my_context) = new_arena->my_default_ctx;
#endif
    }
    // TODO: should it trigger automatic initialization of this thread?
    governor::local_scheduler_weak();
}

void task_arena_base::internal_terminate( ) {
    if( my_arena ) {// task_arena was initialized
        my_arena->my_market->release( /*is_public=*/true, /*blocking_terminate=*/false );
        my_arena->on_thread_leaving<arena::ref_external>();
        my_arena = 0;
#if __TBB_TASK_GROUP_CONTEXT
        my_context = 0;
#endif
    }
}

void task_arena_base::internal_attach( ) {
    __TBB_ASSERT(!my_arena, NULL);
    generic_scheduler* s = governor::local_scheduler_if_initialized();
    if( s && s->my_arena ) {
        // There is an active arena to attach to.
        // It's still used by s, so won't be destroyed right away.
        my_arena = s->my_arena;
        __TBB_ASSERT( my_arena->my_references > 0, NULL );
        my_arena->my_references += arena::ref_external;
#if __TBB_TASK_GROUP_CONTEXT
        my_context = my_arena->my_default_ctx;
        my_version_and_traits |= my_context->my_version_and_traits & exact_exception_flag;
#endif
        my_master_slots = my_arena->my_num_reserved_slots;
        my_max_concurrency = my_master_slots + my_arena->my_max_num_workers;
        __TBB_ASSERT(arena::num_arena_slots(my_max_concurrency)==my_arena->my_num_slots, NULL);
        // increases market's ref count for task_arena
        market::global_market( /*is_public=*/true );
    }
}

void task_arena_base::internal_enqueue( task& t, intptr_t prio ) const {
    __TBB_ASSERT(my_arena, NULL);
    generic_scheduler* s = governor::local_scheduler_if_initialized();
    __TBB_ASSERT(s, "Scheduler is not initialized"); // we allocated a task so can expect the scheduler
#if __TBB_TASK_GROUP_CONTEXT
    __TBB_ASSERT(my_arena->my_default_ctx == t.prefix().context, NULL);
    __TBB_ASSERT(!my_arena->my_default_ctx->is_group_execution_cancelled(), // TODO: any better idea?
                 "The task will not be executed because default task_group_context of task_arena is cancelled. Has previously enqueued task thrown an exception?");
#endif
    my_arena->enqueue_task( t, prio, s->my_random );
}

class delegated_task : public task {
    internal::delegate_base & my_delegate;
    concurrent_monitor & my_monitor;
    task * my_root;
    task* execute() __TBB_override {
        generic_scheduler& s = *(generic_scheduler*)prefix().owner;
        __TBB_ASSERT(s.outermost_level(), "expected to be enqueued and received on the outermost level");
        struct outermost_context : internal::no_copy {
            delegated_task * t;
            generic_scheduler & s;
            task * orig_dummy;
            task_group_context * orig_ctx;
            scheduler_properties orig_props;
            outermost_context(delegated_task *_t, generic_scheduler &_s) 
                : t(_t), s(_s), orig_dummy(s.my_dummy_task), orig_props(s.my_properties) {
                __TBB_ASSERT(s.my_innermost_running_task == t, NULL);
#if __TBB_TASK_GROUP_CONTEXT
                orig_ctx = t->prefix().context;
                t->prefix().context = s.my_arena->my_default_ctx;
#endif
                // Mimics outermost master
                s.my_dummy_task = t;
                s.my_properties.type = scheduler_properties::master;
            }
            ~outermost_context() {
#if __TBB_TASK_GROUP_CONTEXT
                // Restore context for sake of registering potential exception
                t->prefix().context = orig_ctx;
#endif
                s.my_properties = orig_props;
                s.my_dummy_task = orig_dummy;
            }
        } scope(this, s);
        my_delegate();
        return NULL;
    }
    ~delegated_task() {
        // potential exception was already registered. It must happen before the notification
        __TBB_ASSERT(my_root->ref_count()==2, NULL);
        __TBB_store_with_release(my_root->prefix().ref_count, 1); // must precede the wakeup
        my_monitor.notify(*this); // do not relax, it needs a fence!
    }
public:
    delegated_task( internal::delegate_base & d, concurrent_monitor & s, task * t )
        : my_delegate(d), my_monitor(s), my_root(t) {}
    // predicate for concurrent_monitor notification
    bool operator()(uintptr_t ctx) const { return (void*)ctx == (void*)&my_delegate; }
};

void task_arena_base::internal_execute(internal::delegate_base& d) const {
    __TBB_ASSERT(my_arena, NULL);
    generic_scheduler* s = governor::local_scheduler_weak();
    __TBB_ASSERT(s, "Scheduler is not initialized");

    bool same_arena = s->my_arena == my_arena;
    size_t index1 = s->my_arena_index;
    if (!same_arena) {
        index1 = my_arena->occupy_free_slot</* as_worker*/false>(*s);
        if (index1 == arena::out_of_arena) {

#if __TBB_USE_OPTIONAL_RTTI
            // Workaround for the bug inside graph. If the thread can not occupy arena slot during task_arena::execute()
            // and all aggregator operations depend on this task completion (all other threads are inside arena already)
            // deadlock appears, because enqueued task will never enter arena.
            // Workaround: check if the task came from graph via RTTI (casting to graph::spawn_functor)
            // and enqueue this task with non-blocking internal_enqueue method.
            // TODO: have to change behaviour later in next GOLD release (maybe to add new library entry point - try_execute)
            typedef tbb::flow::interface10::graph::spawn_functor graph_funct;
            internal::delegated_function< graph_funct, void >* deleg_funct =
                    dynamic_cast< internal::delegated_function< graph_funct, void>* >(&d);

            if (deleg_funct) {
                internal_enqueue(*new(task::allocate_root(*my_context)) 
                    internal::function_task< internal::strip< graph_funct >::type >
                        (internal::forward< graph_funct >(deleg_funct->my_func)), 0);
                return;
            } else {
#endif
                concurrent_monitor::thread_context waiter;
#if __TBB_TASK_GROUP_CONTEXT
                task_group_context exec_context(task_group_context::isolated, my_version_and_traits & exact_exception_flag);
#if __TBB_FP_CONTEXT
                exec_context.copy_fp_settings(*my_context);
#endif
#endif
                auto_empty_task root(__TBB_CONTEXT_ARG(s, &exec_context));
                root.prefix().ref_count = 2;
                my_arena->enqueue_task(*new(task::allocate_root(__TBB_CONTEXT_ARG1(exec_context)))
                    delegated_task(d, my_arena->my_exit_monitors, &root),
                    0, s->my_random); // TODO: priority?
                size_t index2 = arena::out_of_arena;
                do {
                    my_arena->my_exit_monitors.prepare_wait(waiter, (uintptr_t)&d);
                    if (__TBB_load_with_acquire(root.prefix().ref_count) < 2) {
                        my_arena->my_exit_monitors.cancel_wait(waiter);
                        break;
                    }
                    index2 = my_arena->occupy_free_slot</*as_worker*/false>(*s);
                    if (index2 != arena::out_of_arena) {
                        my_arena->my_exit_monitors.cancel_wait(waiter);
                        nested_arena_context scope(s, my_arena, index2, scheduler_properties::master, same_arena);
                        s->local_wait_for_all(root, NULL);
#if TBB_USE_EXCEPTIONS
                        __TBB_ASSERT(!exec_context.my_exception, NULL); // exception can be thrown above, not deferred
#endif
                        __TBB_ASSERT(root.prefix().ref_count == 0, NULL);
                        break;
                    }
                    my_arena->my_exit_monitors.commit_wait(waiter);
                } while (__TBB_load_with_acquire(root.prefix().ref_count) == 2);
                if (index2 == arena::out_of_arena) {
                    // notify a waiting thread even if this thread did not enter arena,
                    // in case it was woken by a leaving thread but did not need to enter
                    my_arena->my_exit_monitors.notify_one(); // do not relax!
                }
#if TBB_USE_EXCEPTIONS
                // process possible exception
                if (task_group_context::exception_container_type *pe = exec_context.my_exception)
                    TbbRethrowException(pe);
#endif
                return;
#if __TBB_USE_OPTIONAL_RTTI
            } // if task came from graph
#endif
        } // if (index1 == arena::out_of_arena)
    } // if (!same_arena)

    cpu_ctl_env_helper cpu_ctl_helper;
    cpu_ctl_helper.set_env(__TBB_CONTEXT_ARG1(my_context));
#if TBB_USE_EXCEPTIONS
    try {
#endif
        //TODO: replace dummy tasks for workers as well to avoid using of the_dummy_context
        nested_arena_context scope(s, my_arena, index1, scheduler_properties::master, same_arena);
        d();
#if TBB_USE_EXCEPTIONS
    }
    catch (...) {
        cpu_ctl_helper.restore_default(); // TODO: is it needed on Windows?
        if (my_version_and_traits & exact_exception_flag) throw;
        else {
            task_group_context exception_container(task_group_context::isolated,
                task_group_context::default_traits & ~task_group_context::exact_exception);
            exception_container.register_pending_exception();
            __TBB_ASSERT(exception_container.my_exception, NULL);
            TbbRethrowException(exception_container.my_exception);
        }
    }
#endif
}

// this wait task is a temporary approach to wait for arena emptiness for masters without slots
// TODO: it will be rather reworked for one source of notification from is_out_of_work
class wait_task : public task {
    binary_semaphore & my_signal;
    task* execute() __TBB_override {
        generic_scheduler* s = governor::local_scheduler_if_initialized();
        __TBB_ASSERT( s, NULL );
        __TBB_ASSERT( s->outermost_level(), "The enqueued task can be processed only on outermost level" );
        if ( s->is_worker() ) {
            __TBB_ASSERT( s->my_innermost_running_task == this, NULL );
            // Mimic worker on outermost level to run remaining tasks
            s->my_innermost_running_task = s->my_dummy_task;
            s->local_wait_for_all( *s->my_dummy_task, NULL );
            s->my_innermost_running_task = this;
        } else s->my_arena->is_out_of_work(); // avoids starvation of internal_wait: issuing this task makes arena full
        my_signal.V();
        return NULL;
    }
public:
    wait_task ( binary_semaphore & sema ) : my_signal(sema) {}
};

void task_arena_base::internal_wait() const {
    __TBB_ASSERT(my_arena, NULL);
    generic_scheduler* s = governor::local_scheduler_weak();
    __TBB_ASSERT(s, "Scheduler is not initialized");
    __TBB_ASSERT(s->my_arena != my_arena || s->my_arena_index == 0, "task_arena::wait_until_empty() is not supported within a worker context" );
    if( s->my_arena == my_arena ) {
        //unsupported, but try do something for outermost master
        __TBB_ASSERT(s->master_outermost_level(), "unsupported");
        if( !s->my_arena_index )
            while( my_arena->num_workers_active() )
                s->wait_until_empty();
    } else for(;;) {
        while( my_arena->my_pool_state != arena::SNAPSHOT_EMPTY ) {
            if( !__TBB_load_with_acquire(my_arena->my_slots[0].my_scheduler) // TODO TEMP: one master, make more masters
                && as_atomic(my_arena->my_slots[0].my_scheduler).compare_and_swap(s, NULL) == NULL ) {
                nested_arena_context a(s, my_arena, 0, scheduler_properties::worker, false);
                s->wait_until_empty();
            } else {
                binary_semaphore waiter; // TODO: replace by a single event notification from is_out_of_work
                internal_enqueue( *new( task::allocate_root(__TBB_CONTEXT_ARG1(*my_context)) ) wait_task(waiter), 0 ); // TODO: priority?
                waiter.P(); // TODO: concurrent_monitor
            }
        }
        if( !my_arena->num_workers_active() && !my_arena->my_slots[0].my_scheduler) // no activity
            break; // spin until workers active but avoid spinning in a worker
        __TBB_Yield(); // wait until workers and master leave
    }
}

/*static*/ int task_arena_base::internal_current_slot() {
    generic_scheduler* s = governor::local_scheduler_if_initialized();
    return s? int(s->my_arena_index) : -1;
}

#if __TBB_TASK_ISOLATION
class isolation_guard : tbb::internal::no_copy {
    isolation_tag &guarded;
    isolation_tag previous_value;
public:
    isolation_guard( isolation_tag &isolation ) : guarded( isolation ), previous_value( isolation ) {}
    ~isolation_guard() {
        guarded = previous_value;
    }
};

void isolate_within_arena( delegate_base& d, intptr_t reserved ) {
    __TBB_ASSERT( reserved == 0, NULL );
    // TODO: Decide what to do if the scheduler is not initialized. Is there a use case for it?
    generic_scheduler* s = governor::local_scheduler_weak();
    __TBB_ASSERT( s, "this_task_arena::isolate() needs an initialized scheduler" );
    // Theoretically, we can keep the current isolation in the scheduler; however, it makes sense to store it in innermost
    // running task because it can in principle be queried via task::self().
    isolation_tag& current_isolation = s->my_innermost_running_task->prefix().isolation;
    // We temporarily change the isolation tag of the currently running task. It will be restored in the destructor of the guard.
    isolation_guard guard( current_isolation );
    current_isolation = reinterpret_cast<isolation_tag>(&d);
    d();
}
#endif /* __TBB_TASK_ISOLATION */

int task_arena_base::internal_max_concurrency(const task_arena *ta) {
    arena* a = NULL;
    if( ta ) // for special cases of ta->max_concurrency()
        a = ta->my_arena;
    else if( generic_scheduler* s = governor::local_scheduler_if_initialized() )
        a = s->my_arena; // the current arena if any

    if( a ) { // Get parameters from the arena
        __TBB_ASSERT( !ta || ta->my_max_concurrency==1, NULL );
        return a->my_num_reserved_slots + a->my_max_num_workers;
    } else {
        __TBB_ASSERT( !ta || ta->my_max_concurrency==automatic, NULL );
        return int(governor::default_num_threads());
    }
}
} // tbb::interfaceX::internal
} // tbb::interfaceX
} // tbb
