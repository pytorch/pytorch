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

#ifndef _TBB_arena_H
#define _TBB_arena_H

#include "tbb/tbb_stddef.h"
#include "tbb/atomic.h"

#include "tbb/tbb_machine.h"

#include "scheduler_common.h"
#include "intrusive_list.h"
#include "task_stream.h"
#include "../rml/include/rml_tbb.h"
#include "mailbox.h"
#include "observer_proxy.h"
#include "market.h"
#include "governor.h"
#include "concurrent_monitor.h"

namespace tbb {

class task_group_context;
class allocate_root_with_context_proxy;

namespace internal {

//! The structure of an arena, except the array of slots.
/** Separated in order to simplify padding.
    Intrusive list node base class is used by market to form a list of arenas. **/
struct arena_base : padded<intrusive_list_node> {
    //! The number of workers that have been marked out by the resource manager to service the arena.
    unsigned my_num_workers_allotted;   // heavy use in stealing loop

    //! Reference counter for the arena.
    /** Worker and master references are counted separately: first several bits are for references
        from master threads or explicit task_arenas (see arena::ref_external_bits below);
        the rest counts the number of workers servicing the arena. */
    atomic<unsigned> my_references;     // heavy use in stealing loop

#if __TBB_TASK_PRIORITY
    //! The highest priority of recently spawned or enqueued tasks.
    volatile intptr_t my_top_priority;  // heavy use in stealing loop
#endif /* !__TBB_TASK_PRIORITY */

    //! The maximal number of currently busy slots.
    atomic<unsigned> my_limit;          // heavy use in stealing loop

    //! Task pool for the tasks scheduled via task::enqueue() method.
    /** Such scheduling guarantees eventual execution even if
        - new tasks are constantly coming (by extracting scheduled tasks in
          relaxed FIFO order);
        - the enqueuing thread does not call any of wait_for_all methods.
        Depending on __TBB_TASK_PRIORITY, num_priority_levels can be 1 or more. **/
    task_stream<num_priority_levels> my_task_stream; // heavy use in stealing loop

    //! The number of workers requested by the master thread owning the arena.
    unsigned my_max_num_workers;

    //! The number of workers that are currently requested from the resource manager.
    int my_num_workers_requested;

    //! Current task pool state and estimate of available tasks amount.
    /** The estimate is either 0 (SNAPSHOT_EMPTY) or infinity (SNAPSHOT_FULL).
        Special state is "busy" (any other unsigned value).
        Note that the implementation of arena::is_busy_or_empty() requires
        my_pool_state to be unsigned. */
    tbb::atomic<uintptr_t> my_pool_state;

#if __TBB_ARENA_OBSERVER
    //! The list of local observers attached to this arena.
    observer_list my_observers;
#endif

#if __TBB_TASK_PRIORITY
    //! The lowest normalized priority of available spawned or enqueued tasks.
    intptr_t my_bottom_priority;

    //! Tracks events that may bring tasks in offload areas to the top priority level.
    /** Incremented when arena top priority changes or a task group priority
        is elevated to the current arena's top level. **/
    uintptr_t my_reload_epoch;

    //! The list of offloaded tasks abandoned by workers revoked by the market.
    task* my_orphaned_tasks;

    //! Counter used to track the occurrence of recent orphaning and re-sharing operations.
    tbb::atomic<uintptr_t> my_abandonment_epoch;

    //! The highest priority level containing enqueued tasks.
    /** It being greater than 0 means that high priority enqueued tasks had to be
        bypassed because all workers were blocked in nested dispatch loops and
        were unable to progress at then current priority level. **/
    tbb::atomic<intptr_t> my_skipped_fifo_priority;
#endif /* !__TBB_TASK_PRIORITY */

    // Below are rarely modified members

    //! The market that owns this arena.
    market* my_market;

    //! ABA prevention marker.
    uintptr_t my_aba_epoch;

#if !__TBB_FP_CONTEXT
    //! FPU control settings of arena's master thread captured at the moment of arena instantiation.
    cpu_ctl_env my_cpu_ctl_env;
#endif

#if __TBB_TASK_GROUP_CONTEXT
    //! Default task group context.
    /** Used by root tasks allocated directly by the master thread (not from inside
        a TBB task) without explicit context specification. **/
    task_group_context* my_default_ctx;
#endif /* __TBB_TASK_GROUP_CONTEXT */

    //! The number of slots in the arena.
    unsigned my_num_slots;

    //! The number of reserved slots (can be occupied only by masters).
    unsigned my_num_reserved_slots;

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    //! Possible states for the concurrency mode of an arena.
    enum concurrency_mode {
        cm_normal =  0,     // arena is served by workers as usual
        cm_enforced_local,  // arena needs an extra worker despite the arena limit
        cm_enforced_global  // arena needs an extra worker despite a global limit
    };

    //! The concurrency mode of an arena.
    concurrency_mode my_concurrency_mode;
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */

    //! Waiting object for master threads that cannot join the arena.
    concurrent_monitor my_exit_monitors;

#if TBB_USE_ASSERT
    //! Used to trap accesses to the object after its destruction.
    uintptr_t my_guard;
#endif /* TBB_USE_ASSERT */
}; // struct arena_base

class arena: public padded<arena_base>
{
    //! If enqueued tasks found, restore arena priority and task presence status
    void restore_priority_if_need();
public:
    typedef padded<arena_base> base_type;

    //! Types of work advertised by advertise_new_work()
    enum new_work_type {
        work_spawned,
        wakeup,
        work_enqueued
    };

    //! Constructor
    arena ( market&, unsigned max_num_workers, unsigned num_reserved_slots );

    //! Allocate an instance of arena.
    static arena& allocate_arena( market&, unsigned num_slots, unsigned num_reserved_slots );

    static int unsigned num_arena_slots ( unsigned num_slots ) {
        return max(2u, num_slots);
    }

    static int allocation_size ( unsigned num_slots ) {
        return sizeof(base_type) + num_slots * (sizeof(mail_outbox) + sizeof(arena_slot));
    }

    //! Get reference to mailbox corresponding to given affinity_id.
    mail_outbox& mailbox( affinity_id id ) {
        __TBB_ASSERT( 0<id, "affinity id must be positive integer" );
        __TBB_ASSERT( id <= my_num_slots, "affinity id out of bounds" );

        return ((mail_outbox*)this)[-(int)id];
    }

    //! Completes arena shutdown, destructs and deallocates it.
    void free_arena ();

    typedef uintptr_t pool_state_t;

    //! No tasks to steal since last snapshot was taken
    static const pool_state_t SNAPSHOT_EMPTY = 0;

    //! At least one task has been offered for stealing since the last snapshot started
    static const pool_state_t SNAPSHOT_FULL = pool_state_t(-1);

    //! The number of least significant bits for external references
    static const unsigned ref_external_bits = 12; // up to 4095 external and 1M workers

    //! Reference increment values for externals and workers
    static const unsigned ref_external = 1;
    static const unsigned ref_worker   = 1<<ref_external_bits;

    //! No tasks to steal or snapshot is being taken.
    static bool is_busy_or_empty( pool_state_t s ) { return s < SNAPSHOT_FULL; }

    //! The number of workers active in the arena.
    unsigned num_workers_active( ) {
        return my_references >> ref_external_bits;
    }

    //! If necessary, raise a flag that there is new job in arena.
    template<arena::new_work_type work_type> void advertise_new_work();

    //! Check if there is job anywhere in arena.
    /** Return true if no job or if arena is being cleaned up. */
    bool is_out_of_work();

    //! enqueue a task into starvation-resistance queue
    void enqueue_task( task&, intptr_t, FastRandom & );

    //! Registers the worker with the arena and enters TBB scheduler dispatch loop
    void process( generic_scheduler& );

    //! Notification that worker or master leaves its arena
    template<unsigned ref_param>
    inline void on_thread_leaving ( );

#if __TBB_STATISTICS
    //! Outputs internal statistics accumulated by the arena
    void dump_arena_statistics ();
#endif /* __TBB_STATISTICS */

#if __TBB_TASK_PRIORITY
    //! Check if recent priority changes may bring some tasks to the current priority level soon
    /** /param tasks_present indicates presence of tasks at any priority level. **/
    inline bool may_have_tasks ( generic_scheduler*, bool& tasks_present, bool& dequeuing_possible );

    //! Puts offloaded tasks into global list of orphaned tasks
    void orphan_offloaded_tasks ( generic_scheduler& s );
#endif /* __TBB_TASK_PRIORITY */

#if __TBB_COUNT_TASK_NODES
    //! Returns the number of task objects "living" in worker threads
    intptr_t workers_task_node_count();
#endif

    //! Check for the presence of enqueued tasks at all priority levels
    bool has_enqueued_tasks();

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    //! Recall worker if global mandatory is enabled, but not for this arena
    bool recall_by_mandatory_request() const {
        return my_market->my_mandatory_num_requested && my_concurrency_mode==cm_normal;
    }

    //! The arena is currently in an enforced concurrency mode
    bool must_have_concurrency() const {
        return my_num_workers_requested &&
               ( my_concurrency_mode==cm_enforced_local || my_concurrency_mode==cm_enforced_global );
    }
#endif
    static const size_t out_of_arena = ~size_t(0);
    //! Tries to occupy a slot in the arena. On success, returns the slot index; if no slot is available, returns out_of_arena.
    template <bool as_worker>
    size_t occupy_free_slot( generic_scheduler& s );
    //! Tries to occupy a slot in the specified range.
    size_t occupy_free_slot_in_range( generic_scheduler& s, size_t lower, size_t upper );

    /** Must be the last data field */
    arena_slot my_slots[1];
}; // class arena

template<unsigned ref_param>
inline void arena::on_thread_leaving ( ) {
    //
    // Implementation of arena destruction synchronization logic contained various
    // bugs/flaws at the different stages of its evolution, so below is a detailed
    // description of the issues taken into consideration in the framework of the
    // current design.
    //
    // In case of using fire-and-forget tasks (scheduled via task::enqueue())
    // master thread is allowed to leave its arena before all its work is executed,
    // and market may temporarily revoke all workers from this arena. Since revoked
    // workers never attempt to reset arena state to EMPTY and cancel its request
    // to RML for threads, the arena object is destroyed only when both the last
    // thread is leaving it and arena's state is EMPTY (that is its master thread
    // left and it does not contain any work).
    // Thus resetting arena to EMPTY state (as earlier TBB versions did) should not
    // be done here (or anywhere else in the master thread to that matter); doing so
    // can result either in arena's premature destruction (at least without
    // additional costly checks in workers) or in unnecessary arena state changes
    // (and ensuing workers migration).
    //
    // A worker that checks for work presence and transitions arena to the EMPTY
    // state (in snapshot taking procedure arena::is_out_of_work()) updates
    // arena::my_pool_state first and only then arena::my_num_workers_requested.
    // So the check for work absence must be done against the latter field.
    //
    // In a time window between decrementing the active threads count and checking
    // if there is an outstanding request for workers. New worker thread may arrive,
    // finish remaining work, set arena state to empty, and leave decrementing its
    // refcount and destroying. Then the current thread will destroy the arena
    // the second time. To preclude it a local copy of the outstanding request
    // value can be stored before decrementing active threads count.
    //
    // But this technique may cause two other problem. When the stored request is
    // zero, it is possible that arena still has threads and they can generate new
    // tasks and thus re-establish non-zero requests. Then all the threads can be
    // revoked (as described above) leaving this thread the last one, and causing
    // it to destroy non-empty arena.
    //
    // The other problem takes place when the stored request is non-zero. Another
    // thread may complete the work, set arena state to empty, and leave without
    // arena destruction before this thread decrements the refcount. This thread
    // cannot destroy the arena either. Thus the arena may be "orphaned".
    //
    // In both cases we cannot dereference arena pointer after the refcount is
    // decremented, as our arena may already be destroyed.
    //
    // If this is the master thread, the market is protected by refcount to it.
    // In case of workers market's liveness is ensured by the RML connection
    // rundown protocol, according to which the client (i.e. the market) lives
    // until RML server notifies it about connection termination, and this
    // notification is fired only after all workers return into RML.
    //
    // Thus if we decremented refcount to zero we ask the market to check arena
    // state (including the fact if it is alive) under the lock.
    //
    uintptr_t aba_epoch = my_aba_epoch;
    market* m = my_market;
    __TBB_ASSERT(my_references >= ref_param, "broken arena reference counter");
#if __TBB_STATISTICS_EARLY_DUMP
    // While still holding a reference to the arena, compute how many external references are left.
    // If just one, dump statistics.
    if ( modulo_power_of_two(my_references,ref_worker)==ref_param ) // may only be true with ref_external
        GATHER_STATISTIC( dump_arena_statistics() );
#endif
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    // When there is no workers someone must free arena, as
    // without workers, no one calls is_out_of_work().
    // Skip workerless arenas because they have no demand for workers.
    // TODO: consider more strict conditions for the cleanup,
    // because it can create the demand of workers,
    // but the arena can be already empty (and so ready for destroying)
    if( ref_param==ref_external && my_num_slots != my_num_reserved_slots
        && 0 == m->my_num_workers_soft_limit && my_concurrency_mode==cm_normal ) {
        bool is_out = false;
        for (int i=0; i<num_priority_levels; i++) {
            is_out = is_out_of_work();
            if (is_out)
                break;
        }
        // We expect, that in worst case it's enough to have num_priority_levels-1
        // calls to restore priorities and and yet another is_out_of_work() to conform
        // that no work was found. But as market::set_active_num_workers() can be called
        // concurrently, can't guarantee last is_out_of_work() return true.
    }
#endif
    if ( (my_references -= ref_param ) == 0 )
        m->try_destroy_arena( this, aba_epoch );
}

template<arena::new_work_type work_type> void arena::advertise_new_work() {
    if( work_type == work_enqueued ) {
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        if( my_market->my_num_workers_soft_limit == 0 ) {
            if( my_concurrency_mode!=cm_enforced_global ) {
                if( my_market->mandatory_concurrency_enable( this ) ) {
                    my_pool_state = SNAPSHOT_FULL;
                    return;
                }
            }
        } else if( my_max_num_workers==0 && my_num_reserved_slots==1 ) {
            my_max_num_workers = 1;
            __TBB_ASSERT(my_concurrency_mode==cm_normal, NULL);
            my_concurrency_mode = cm_enforced_local;
            my_pool_state = SNAPSHOT_FULL;
            my_market->adjust_demand( *this, 1 );
            return;
        }
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */
        // Local memory fence here and below is required to avoid missed wakeups; see the comment below.
        // Starvation resistant tasks require concurrency, so missed wakeups are unacceptable.
        atomic_fence();
    }
    else if( work_type == wakeup ) {
        __TBB_ASSERT(my_max_num_workers!=0, "Unexpected worker wakeup request");
        atomic_fence();
    }
    // Double-check idiom that, in case of spawning, is deliberately sloppy about memory fences.
    // Technically, to avoid missed wakeups, there should be a full memory fence between the point we
    // released the task pool (i.e. spawned task) and read the arena's state.  However, adding such a
    // fence might hurt overall performance more than it helps, because the fence would be executed
    // on every task pool release, even when stealing does not occur.  Since TBB allows parallelism,
    // but never promises parallelism, the missed wakeup is not a correctness problem.
    pool_state_t snapshot = my_pool_state;
    if( is_busy_or_empty(snapshot) ) {
        // Attempt to mark as full.  The compare_and_swap below is a little unusual because the
        // result is compared to a value that can be different than the comparand argument.
        if( my_pool_state.compare_and_swap( SNAPSHOT_FULL, snapshot )==SNAPSHOT_EMPTY ) {
            if( snapshot!=SNAPSHOT_EMPTY ) {
                // This thread read "busy" into snapshot, and then another thread transitioned
                // my_pool_state to "empty" in the meantime, which caused the compare_and_swap above
                // to fail.  Attempt to transition my_pool_state from "empty" to "full".
                if( my_pool_state.compare_and_swap( SNAPSHOT_FULL, SNAPSHOT_EMPTY )!=SNAPSHOT_EMPTY ) {
                    // Some other thread transitioned my_pool_state from "empty", and hence became
                    // responsible for waking up workers.
                    return;
                }
            }
            // This thread transitioned pool from empty to full state, and thus is responsible for
            // telling the market that there is work to do.
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
            if( work_type == work_spawned ) {
                if( my_concurrency_mode!=cm_normal ) {
                    switch( my_concurrency_mode ) {
                    case cm_enforced_local:
                        __TBB_ASSERT(my_max_num_workers==1, "");
                        __TBB_ASSERT(!governor::local_scheduler()->is_worker(), "");
                        // There was deliberate oversubscription on 1 core for sake of starvation-resistant tasks.
                        // Now a single active thread (must be the master) supposedly starts a new parallel region
                        // with relaxed sequential semantics, and oversubscription should be avoided.
                        // Demand for workers has been decreased to 0 during SNAPSHOT_EMPTY, so just keep it.
                        my_max_num_workers = 0;
                        my_concurrency_mode = cm_normal;
                        break;
                    case cm_enforced_global:
                        my_market->mandatory_concurrency_disable( this );
                        restore_priority_if_need();
                        break;
                    default:
                        break;
                    }
                    return;
                }
            }
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */
            my_market->adjust_demand( *this, my_max_num_workers );
        }
    }
}

} // namespace internal
} // namespace tbb

#endif /* _TBB_arena_H */
