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

#ifndef _TBB_scheduler_H
#define _TBB_scheduler_H

#include "scheduler_common.h"
#include "tbb/spin_mutex.h"
#include "mailbox.h"
#include "tbb_misc.h" // for FastRandom
#include "itt_notify.h"
#include "../rml/include/rml_tbb.h"

#include "intrusive_list.h"

#if __TBB_SURVIVE_THREAD_SWITCH
#include "cilk-tbb-interop.h"
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

namespace tbb {
namespace internal {

template<typename SchedulerTraits> class custom_scheduler;

//------------------------------------------------------------------------
// generic_scheduler
//------------------------------------------------------------------------

#define EmptyTaskPool ((task**)0)
#define LockedTaskPool ((task**)~(intptr_t)0)

//! Bit-field representing properties of a sheduler
struct scheduler_properties {
    static const bool worker = false;
    static const bool master = true;
    //! Indicates that a scheduler acts as a master or a worker.
    bool type : 1;
    //! Indicates that a scheduler is on outermost level.
    /**  Note that the explicit execute method will set this property. **/
    bool outermost : 1;
    //! Reserved bits
    unsigned char : 6;
};

struct scheduler_state {
    //! Index of the arena slot the scheduler occupies now, or occupied last time.
    size_t my_arena_index; // TODO: make it unsigned and pair with my_affinity_id to fit into cache line

    //! Pointer to the slot in the arena we own at the moment.
    arena_slot* my_arena_slot;

    //! The arena that I own (if master) or am servicing at the moment (if worker)
    arena* my_arena;

    //! Innermost task whose task::execute() is running. A dummy task on the outermost level.
    task* my_innermost_running_task;

    mail_inbox my_inbox;

    //! The mailbox id assigned to this scheduler.
    /** The id is assigned upon first entry into the arena.
        TODO: how are id's being garbage collected?
        TODO: master thread may enter arena and leave and then reenter.
                We want to give it the same affinity_id upon reentry, if practical.
        TODO: investigate if it makes sense to merge this field into scheduler_properties.
      */
    affinity_id my_affinity_id;

    scheduler_properties my_properties;

#if __TBB_SCHEDULER_OBSERVER
    //! Last observer in the global observers list processed by this scheduler
    observer_proxy* my_last_global_observer;
#endif

#if __TBB_ARENA_OBSERVER
    //! Last observer in the local observers list processed by this scheduler
    observer_proxy* my_last_local_observer;
#endif
#if __TBB_TASK_PRIORITY
    //! Latest known highest priority of tasks in the market or arena.
    /** Master threads currently tracks only tasks in their arenas, while workers
        take into account global top priority (among all arenas in the market). **/
    volatile intptr_t *my_ref_top_priority;

    //! Pointer to market's (for workers) or current arena's (for the master) reload epoch counter.
    volatile uintptr_t *my_ref_reload_epoch;
#endif /* __TBB_TASK_PRIORITY */
};

//! Work stealing task scheduler.
/** None of the fields here are ever read or written by threads other than
    the thread that creates the instance.

    Class generic_scheduler is an abstract base class that contains most of the scheduler,
    except for tweaks specific to processors and tools (e.g. VTune).
    The derived template class custom_scheduler<SchedulerTraits> fills in the tweaks. */
class generic_scheduler: public scheduler
                       , public ::rml::job
                       , public intrusive_list_node
                       , public scheduler_state {
public: // almost every class in TBB uses generic_scheduler

    //! If sizeof(task) is <=quick_task_size, it is handled on a free list instead of malloc'd.
    static const size_t quick_task_size = 256-task_prefix_reservation_size;

    static bool is_version_3_task( task& t ) {
        return (t.prefix().extra_state & 0x0F)>=0x1;
    }

    //! Position in the call stack specifying its maximal filling when stealing is still allowed
    uintptr_t my_stealing_threshold;
#if __TBB_ipf
    //! Position in the RSE backup area specifying its maximal filling when stealing is still allowed
    uintptr_t my_rsb_stealing_threshold;
#endif

    static const size_t null_arena_index = ~size_t(0);

    inline bool is_task_pool_published () const;

    inline bool is_local_task_pool_quiescent () const;

    inline bool is_quiescent_local_task_pool_empty () const;

    inline bool is_quiescent_local_task_pool_reset () const;

    //! The market I am in
    market* my_market;

    //! Random number generator used for picking a random victim from which to steal.
    FastRandom my_random;

    //! Free list of small tasks that can be reused.
    task* my_free_list;

#if __TBB_HOARD_NONLOCAL_TASKS
    //! Free list of small non-local tasks that should be returned or can be reused.
    task* my_nonlocal_free_list;
#endif
    //! Fake root task created by slave threads.
    /** The task is used as the "parent" argument to method wait_for_all. */
    task* my_dummy_task;

    //! Reference count for scheduler
    /** Number of task_scheduler_init objects that point to this scheduler */
    long my_ref_count;

    inline void attach_mailbox( affinity_id id );

    /* A couple of bools can be located here because space is otherwise just padding after my_affinity_id. */

    //! True if *this was created by automatic TBB initialization
    bool my_auto_initialized;

#if __TBB_COUNT_TASK_NODES
    //! Net number of big task objects that have been allocated but not yet freed.
    intptr_t my_task_node_count;
#endif /* __TBB_COUNT_TASK_NODES */

    //! Sets up the data necessary for the stealing limiting heuristics
    void init_stack_info ();

    //! Returns true if stealing is allowed
    bool can_steal () {
        int anchor;
        // TODO IDEA: Add performance warning?
#if __TBB_ipf
        return my_stealing_threshold < (uintptr_t)&anchor && (uintptr_t)__TBB_get_bsp() < my_rsb_stealing_threshold;
#else
        return my_stealing_threshold < (uintptr_t)&anchor;
#endif
    }

    //! Used by workers to enter the task pool
    /** Does not lock the task pool in case if arena slot has been successfully grabbed. **/
    void publish_task_pool();

    //! Leave the task pool
    /** Leaving task pool automatically releases the task pool if it is locked. **/
    void leave_task_pool();

    //! Resets head and tail indices to 0, and leaves task pool
    /** The task pool must be locked by the owner (via acquire_task_pool).**/
    inline void reset_task_pool_and_leave ();

    //! Locks victim's task pool, and returns pointer to it. The pointer can be NULL.
    /** Garbles victim_arena_slot->task_pool for the duration of the lock. **/
    task** lock_task_pool( arena_slot* victim_arena_slot ) const;

    //! Unlocks victim's task pool
    /** Restores victim_arena_slot->task_pool munged by lock_task_pool. **/
    void unlock_task_pool( arena_slot* victim_arena_slot, task** victim_task_pool ) const;

    //! Locks the local task pool
    /** Garbles my_arena_slot->task_pool for the duration of the lock. Requires
        correctly set my_arena_slot->task_pool_ptr. **/
    void acquire_task_pool() const;

    //! Unlocks the local task pool
    /** Restores my_arena_slot->task_pool munged by acquire_task_pool. Requires
        correctly set my_arena_slot->task_pool_ptr. **/
    void release_task_pool() const;

    //! Checks if t is affinitized to another thread, and if so, bundles it as proxy.
    /** Returns either t or proxy containing t. **/
    task* prepare_for_spawning( task* t );

    //! Makes newly spawned tasks visible to thieves
    inline void commit_spawned_tasks( size_t new_tail );

    //! Makes relocated tasks visible to thieves and releases the local task pool.
    /** Obviously, the task pool must be locked when calling this method. **/
    inline void commit_relocated_tasks( size_t new_tail );

    //! Get a task from the local pool.
    /** Called only by the pool owner.
        Returns the pointer to the task or NULL if a suitable task is not found.
        Resets the pool if it is empty. **/
    task* get_task( __TBB_ISOLATION_EXPR( isolation_tag isolation ) );

    //! Get a task from the local pool at specified location T.
    /** Returns the pointer to the task or NULL if the task cannot be executed,
        e.g. proxy has been deallocated or isolation constraint is not met.
        tasks_omitted tells if some tasks have been omitted.
        Called only by the pool owner. The caller should guarantee that the
        position T is not available for a thief. **/
#if __TBB_TASK_ISOLATION
    task* get_task( size_t T, isolation_tag isolation, bool& tasks_omitted );
#else
    task* get_task( size_t T );
#endif /* __TBB_TASK_ISOLATION */
    //! Attempt to get a task from the mailbox.
    /** Gets a task only if it has not been executed by its sender or a thief
        that has stolen it from the sender's task pool. Otherwise returns NULL.

        This method is intended to be used only by the thread extracting the proxy
        from its mailbox. (In contrast to local task pool, mailbox can be read only
        by its owner). **/
    task* get_mailbox_task( __TBB_ISOLATION_EXPR( isolation_tag isolation ) );

    //! True if t is a task_proxy
    static bool is_proxy( const task& t ) {
        return t.prefix().extra_state==es_task_proxy;
    }

    //! Steal task from another scheduler's ready pool.
    task* steal_task( __TBB_ISOLATION_ARG( arena_slot& victim_arena_slot, isolation_tag isolation ) );

    /** Initial size of the task deque sufficient to serve without reallocation
        4 nested parallel_for calls with iteration space of 65535 grains each. **/
    static const size_t min_task_pool_size = 64;

    //! Makes sure that the task pool can accommodate at least n more elements
    /** If necessary relocates existing task pointers or grows the ready task deque.
        Returns (possible updated) tail index (not accounting for n). **/
    size_t prepare_task_pool( size_t n );

    //! Initialize a scheduler for a master thread.
    static generic_scheduler* create_master( arena* a );

    //! Perform necessary cleanup when a master thread stops using TBB.
    bool cleanup_master( bool blocking_terminate );

    //! Initialize a scheduler for a worker thread.
    static generic_scheduler* create_worker( market& m, size_t index );

    //! Perform necessary cleanup when a worker thread finishes.
    static void cleanup_worker( void* arg, bool worker );

protected:
    template<typename SchedulerTraits> friend class custom_scheduler;
    generic_scheduler( market & );

public:
#if TBB_USE_ASSERT > 1
    //! Check that internal data structures are in consistent state.
    /** Raises __TBB_ASSERT failure if inconsistency is found. */
    void assert_task_pool_valid() const;
#else
    void assert_task_pool_valid() const {}
#endif /* TBB_USE_ASSERT <= 1 */

    void attach_arena( arena*, size_t index, bool is_master );
    void nested_arena_entry( arena*, size_t );
    void nested_arena_exit();
    void wait_until_empty();

    void spawn( task& first, task*& next ) __TBB_override;

    void spawn_root_and_wait( task& first, task*& next ) __TBB_override;

    void enqueue( task&, void* reserved ) __TBB_override;

    void local_spawn( task* first, task*& next );
    void local_spawn_root_and_wait( task* first, task*& next );
    virtual void local_wait_for_all( task& parent, task* child ) = 0;

    //! Destroy and deallocate this scheduler object
    void free_scheduler();

    //! Allocate task object, either from the heap or a free list.
    /** Returns uninitialized task object with initialized prefix. */
    task& allocate_task( size_t number_of_bytes,
                       __TBB_CONTEXT_ARG(task* parent, task_group_context* context) );

    //! Put task on free list.
    /** Does not call destructor. */
    template<free_task_hint h>
    void free_task( task& t );

    //! Return task object to the memory allocator.
    inline void deallocate_task( task& t );

    //! True if running on a worker thread, false otherwise.
    inline bool is_worker() const;

    //! True if the scheduler is on the outermost dispatch level.
    inline bool outermost_level() const;

    //! True if the scheduler is on the outermost dispatch level in a master thread.
    /** Returns true when this scheduler instance is associated with an application
        thread, and is not executing any TBB task. This includes being in a TBB
        dispatch loop (one of wait_for_all methods) invoked directly from that thread. **/
    inline bool master_outermost_level () const;

    //! True if the scheduler is on the outermost dispatch level in a worker thread.
    inline bool worker_outermost_level () const;

    //! Returns the concurrency limit of the current arena.
    unsigned max_threads_in_arena();

#if __TBB_COUNT_TASK_NODES
    intptr_t get_task_node_count( bool count_arena_workers = false );
#endif /* __TBB_COUNT_TASK_NODES */

    //! Special value used to mark my_return_list as not taking any more entries.
    static task* plugged_return_list() {return (task*)(intptr_t)(-1);}

    //! Number of small tasks that have been allocated by this scheduler.
    __TBB_atomic intptr_t my_small_task_count;

    //! List of small tasks that have been returned to this scheduler by other schedulers.
    // TODO IDEA: see if putting my_return_list on separate cache line improves performance
    task* my_return_list;

    //! Try getting a task from other threads (via mailbox, stealing, FIFO queue, orphans adoption).
    /** Returns obtained task or NULL if all attempts fail. */
    virtual task* receive_or_steal_task( __TBB_ISOLATION_ARG( __TBB_atomic reference_count& completion_ref_count, isolation_tag isolation ) ) = 0;

    //! Free a small task t that that was allocated by a different scheduler
    void free_nonlocal_small_task( task& t );

#if __TBB_TASK_GROUP_CONTEXT
    //! Returns task group context used by this scheduler instance.
    /** This context is associated with root tasks created by a master thread
        without explicitly specified context object outside of any running task.

        Note that the default context of a worker thread is never accessed by
        user code (directly or indirectly). **/
    inline task_group_context* default_context ();

    //! Padding isolating thread-local members from members that can be written to by other threads.
    char _padding1[NFS_MaxLineSize - sizeof(context_list_node_t)];

    //! Head of the thread specific list of task group contexts.
    context_list_node_t my_context_list_head;

    //! Mutex protecting access to the list of task group contexts.
    // TODO: check whether it can be deadly preempted and replace by spinning/sleeping mutex
    spin_mutex my_context_list_mutex;

    //! Last state propagation epoch known to this thread
    /** Together with the_context_state_propagation_epoch constitute synchronization protocol
        that keeps hot path of task group context construction destruction mostly
        lock-free.
        When local epoch equals the global one, the state of task group contexts
        registered with this thread is consistent with that of the task group trees
        they belong to. **/
    uintptr_t my_context_state_propagation_epoch;

    //! Flag indicating that a context is being destructed by its owner thread
    /** Together with my_nonlocal_ctx_list_update constitute synchronization protocol
        that keeps hot path of context destruction (by the owner thread) mostly
        lock-free. **/
    tbb::atomic<uintptr_t> my_local_ctx_list_update;

#if __TBB_TASK_PRIORITY
    //! Returns reference priority used to decide whether a task should be offloaded.
    inline intptr_t effective_reference_priority () const;

    // TODO: move into slots and fix is_out_of_work
    //! Task pool for offloading tasks with priorities lower than the current top priority.
    task* my_offloaded_tasks;

    //! Points to the last offloaded task in the my_offloaded_tasks list.
    task** my_offloaded_task_list_tail_link;

    //! Indicator of how recently the offload area was checked for the presence of top priority tasks.
    uintptr_t my_local_reload_epoch;

    //! Indicates that the pool is likely non-empty even if appears so from outside
    volatile bool my_pool_reshuffling_pending;

    //! Searches offload area for top priority tasks and reloads found ones into primary task pool.
    /** Returns one of the found tasks or NULL. **/
    task* reload_tasks( __TBB_ISOLATION_EXPR( isolation_tag isolation ) );

    task* reload_tasks( task*& offloaded_tasks, task**& offloaded_task_list_link, __TBB_ISOLATION_ARG( intptr_t top_priority, isolation_tag isolation ) );

    //! Moves tasks with priority below the top one from primary task pool into offload area.
    /** Returns the next execution candidate task or NULL. **/
    task* winnow_task_pool ( __TBB_ISOLATION_EXPR( isolation_tag isolation ) );

    //! Get a task from locked or empty pool in range [H0, T0). Releases or unlocks the task pool.
    /** Returns the found task or NULL. **/
    task *get_task_and_activate_task_pool( size_t H0 , __TBB_ISOLATION_ARG( size_t T0, isolation_tag isolation ) );

    //! Unconditionally moves the task into offload area.
    inline void offload_task ( task& t, intptr_t task_priority );
#endif /* __TBB_TASK_PRIORITY */

    //! Detaches abandoned contexts
    /** These contexts must be destroyed by other threads. **/
    void cleanup_local_context_list ();

    //! Finds all contexts registered by this scheduler affected by the state change
    //! and propagates the new state to them.
    template <typename T>
    void propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state );

    // check consistency
    static void assert_context_valid(const task_group_context *tgc) {
        suppress_unused_warning(tgc);
#if TBB_USE_ASSERT
        __TBB_ASSERT(tgc, NULL);
        uintptr_t ctx = tgc->my_version_and_traits;
        __TBB_ASSERT(is_alive(ctx), "referenced task_group_context was destroyed");
        static const char *msg = "task_group_context is invalid";
        __TBB_ASSERT(!(ctx&~(3|(7<<task_group_context::traits_offset))), msg); // the value fits known values of versions and traits
        __TBB_ASSERT(tgc->my_kind < task_group_context::dying, msg);
        __TBB_ASSERT(tgc->my_cancellation_requested == 0 || tgc->my_cancellation_requested == 1, msg);
        __TBB_ASSERT(tgc->my_state < task_group_context::low_unused_state_bit, msg);
        if(tgc->my_kind != task_group_context::isolated) {
            __TBB_ASSERT(tgc->my_owner, msg);
            __TBB_ASSERT(tgc->my_node.my_next && tgc->my_node.my_prev, msg);
        }
#if __TBB_TASK_PRIORITY
        assert_priority_valid(tgc->my_priority);
#endif
        if(tgc->my_parent)
#if TBB_USE_ASSERT > 1
            assert_context_valid(tgc->my_parent);
#else
            __TBB_ASSERT(is_alive(tgc->my_parent->my_version_and_traits), msg);
#endif
#endif
    }
#endif /* __TBB_TASK_GROUP_CONTEXT */

#if _WIN32||_WIN64
private:
    //! Handle returned by RML when registering a master with RML
    ::rml::server::execution_resource_t master_exec_resource;
public:
#endif /* _WIN32||_WIN64 */

#if __TBB_TASK_GROUP_CONTEXT
    //! Flag indicating that a context is being destructed by non-owner thread.
    /** See also my_local_ctx_list_update. **/
    tbb::atomic<uintptr_t> my_nonlocal_ctx_list_update;
#endif /* __TBB_TASK_GROUP_CONTEXT */

#if __TBB_SURVIVE_THREAD_SWITCH
    __cilk_tbb_unwatch_thunk my_cilk_unwatch_thunk;
#if TBB_USE_ASSERT
    //! State values used to check interface contract with cilkrts.
    /** Names of cs_running...cs_freed derived from state machine diagram in cilk-tbb-interop.h */
    enum cilk_state_t {
        cs_none=0xF000, // Start at nonzero value so that we can detect use of zeroed memory.
        cs_running,
        cs_limbo,
        cs_freed
    };
    cilk_state_t my_cilk_state;
#endif /* TBB_USE_ASSERT */
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

#if __TBB_STATISTICS
    //! Set of counters to track internal statistics on per thread basis
    /** Placed at the end of the class definition to minimize the disturbance of
        the core logic memory operations. **/
    mutable statistics_counters my_counters;
#endif /* __TBB_STATISTICS */

}; // class generic_scheduler


} // namespace internal
} // namespace tbb

#include "arena.h"
#include "governor.h"

namespace tbb {
namespace internal {

inline bool generic_scheduler::is_task_pool_published () const {
    __TBB_ASSERT(my_arena_slot, 0);
    return my_arena_slot->task_pool != EmptyTaskPool;
}

inline bool generic_scheduler::is_local_task_pool_quiescent () const {
    __TBB_ASSERT(my_arena_slot, 0);
    task** tp = my_arena_slot->task_pool;
    return tp == EmptyTaskPool || tp == LockedTaskPool;
}

inline bool generic_scheduler::is_quiescent_local_task_pool_empty () const {
    __TBB_ASSERT( is_local_task_pool_quiescent(), "Task pool is not quiescent" );
    return __TBB_load_relaxed(my_arena_slot->head) == __TBB_load_relaxed(my_arena_slot->tail);
}

inline bool generic_scheduler::is_quiescent_local_task_pool_reset () const {
    __TBB_ASSERT( is_local_task_pool_quiescent(), "Task pool is not quiescent" );
    return __TBB_load_relaxed(my_arena_slot->head) == 0 && __TBB_load_relaxed(my_arena_slot->tail) == 0;
}

inline bool generic_scheduler::outermost_level () const {
    return my_properties.outermost;
}

inline bool generic_scheduler::master_outermost_level () const {
    return !is_worker() && outermost_level();
}

inline bool generic_scheduler::worker_outermost_level () const {
    return is_worker() && outermost_level();
}

#if __TBB_TASK_GROUP_CONTEXT
inline task_group_context* generic_scheduler::default_context () {
    return my_dummy_task->prefix().context;
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

inline void generic_scheduler::attach_mailbox( affinity_id id ) {
    __TBB_ASSERT(id>0,NULL);
    my_inbox.attach( my_arena->mailbox(id) );
    my_affinity_id = id;
}

inline bool generic_scheduler::is_worker() const {
    return my_properties.type == scheduler_properties::worker;
}

inline unsigned generic_scheduler::max_threads_in_arena() {
    __TBB_ASSERT(my_arena, NULL);
    return my_arena->my_num_slots;
}

//! Return task object to the memory allocator.
inline void generic_scheduler::deallocate_task( task& t ) {
#if TBB_USE_ASSERT
    task_prefix& p = t.prefix();
    p.state = 0xFF;
    p.extra_state = 0xFF;
    poison_pointer(p.next);
#endif /* TBB_USE_ASSERT */
    NFS_Free((char*)&t-task_prefix_reservation_size);
#if __TBB_COUNT_TASK_NODES
    --my_task_node_count;
#endif /* __TBB_COUNT_TASK_NODES */
}

#if __TBB_COUNT_TASK_NODES
inline intptr_t generic_scheduler::get_task_node_count( bool count_arena_workers ) {
    return my_task_node_count + (count_arena_workers? my_arena->workers_task_node_count(): 0);
}
#endif /* __TBB_COUNT_TASK_NODES */

inline void generic_scheduler::reset_task_pool_and_leave () {
    __TBB_ASSERT( my_arena_slot->task_pool == LockedTaskPool, "Task pool must be locked when resetting task pool" );
    __TBB_store_relaxed( my_arena_slot->tail, 0 );
    __TBB_store_relaxed( my_arena_slot->head, 0 );
    leave_task_pool();
}

//TODO: move to arena_slot
inline void generic_scheduler::commit_spawned_tasks( size_t new_tail ) {
    __TBB_ASSERT ( new_tail <= my_arena_slot->my_task_pool_size, "task deque end was overwritten" );
    // emit "task was released" signal
    ITT_NOTIFY(sync_releasing, (void*)((uintptr_t)my_arena_slot+sizeof(uintptr_t)));
    // Release fence is necessary to make sure that previously stored task pointers
    // are visible to thieves.
    __TBB_store_with_release( my_arena_slot->tail, new_tail );
}

void generic_scheduler::commit_relocated_tasks ( size_t new_tail ) {
    __TBB_ASSERT( is_local_task_pool_quiescent(),
                  "Task pool must be locked when calling commit_relocated_tasks()" );
    __TBB_store_relaxed( my_arena_slot->head, 0 );
    // Tail is updated last to minimize probability of a thread making arena
    // snapshot being misguided into thinking that this task pool is empty.
    __TBB_store_release( my_arena_slot->tail, new_tail );
    release_task_pool();
}

template<free_task_hint hint>
void generic_scheduler::free_task( task& t ) {
#if __TBB_HOARD_NONLOCAL_TASKS
    static const int h = hint&(~local_task);
#else
    static const free_task_hint h = hint;
#endif
    GATHER_STATISTIC(--my_counters.active_tasks);
    task_prefix& p = t.prefix();
    // Verify that optimization hints are correct.
    __TBB_ASSERT( h!=small_local_task || p.origin==this, NULL );
    __TBB_ASSERT( !(h&small_task) || p.origin, NULL );
    __TBB_ASSERT( !(h&local_task) || (!p.origin || uintptr_t(p.origin) > uintptr_t(4096)), "local_task means allocated");
    poison_value(p.depth);
    poison_value(p.ref_count);
    poison_pointer(p.owner);
    __TBB_ASSERT( 1L<<t.state() & (1L<<task::executing|1L<<task::allocated), NULL );
    p.state = task::freed;
    if( h==small_local_task || p.origin==this ) {
        GATHER_STATISTIC(++my_counters.free_list_length);
        p.next = my_free_list;
        my_free_list = &t;
    } else if( !(h&local_task) && p.origin && uintptr_t(p.origin) < uintptr_t(4096) ) {
        // a special value reserved for future use, do nothing since
        // origin is not pointing to a scheduler instance
    } else if( !(h&local_task) && p.origin ) {
        GATHER_STATISTIC(++my_counters.free_list_length);
#if __TBB_HOARD_NONLOCAL_TASKS
        if( !(h&no_cache) ) {
            p.next = my_nonlocal_free_list;
            my_nonlocal_free_list = &t;
        } else
#endif
        free_nonlocal_small_task(t);
    } else {
        GATHER_STATISTIC(--my_counters.big_tasks);
        deallocate_task(t);
    }
}

#if __TBB_TASK_PRIORITY
inline intptr_t generic_scheduler::effective_reference_priority () const {
    // Workers on the outermost dispatch level (i.e. with empty stack) use market's
    // priority as a reference point (to speedup discovering process level priority
    // changes). But when there are enough workers to service (even if only partially)
    // a lower priority arena, they should use arena's priority as a reference, lest
    // be trapped in a futile spinning (because market's priority would prohibit
    // executing ANY tasks in this arena).
    return !worker_outermost_level() ||
        (my_arena->my_num_workers_allotted < my_arena->num_workers_active()
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
         && my_arena->my_concurrency_mode!=arena_base::cm_enforced_global
#endif
            ) ? *my_ref_top_priority : my_arena->my_top_priority;
}

inline void generic_scheduler::offload_task ( task& t, intptr_t /*priority*/ ) {
    GATHER_STATISTIC( ++my_counters.prio_tasks_offloaded );
    __TBB_ASSERT( !is_proxy(t), "The proxy task cannot be offloaded" );
    __TBB_ASSERT( my_offloaded_task_list_tail_link && !*my_offloaded_task_list_tail_link, NULL );
#if TBB_USE_ASSERT
    t.prefix().state = task::ready;
#endif /* TBB_USE_ASSERT */
    t.prefix().next_offloaded = my_offloaded_tasks;
    my_offloaded_tasks = &t;
}
#endif /* __TBB_TASK_PRIORITY */

#if __TBB_FP_CONTEXT
class cpu_ctl_env_helper {
    cpu_ctl_env guard_cpu_ctl_env;
    cpu_ctl_env curr_cpu_ctl_env;
public:
    cpu_ctl_env_helper() {
        guard_cpu_ctl_env.get_env();
        curr_cpu_ctl_env = guard_cpu_ctl_env;
    }
    ~cpu_ctl_env_helper() {
        if ( curr_cpu_ctl_env != guard_cpu_ctl_env )
            guard_cpu_ctl_env.set_env();
    }
    void set_env( const task_group_context *ctx ) {
        generic_scheduler::assert_context_valid(ctx);
        const cpu_ctl_env &ctl = *punned_cast<cpu_ctl_env*>(&ctx->my_cpu_ctl_env);
        if ( ctl != curr_cpu_ctl_env ) {
            curr_cpu_ctl_env = ctl;
            curr_cpu_ctl_env.set_env();
        }
    }
    void restore_default() {
        if ( curr_cpu_ctl_env != guard_cpu_ctl_env ) {
            guard_cpu_ctl_env.set_env();
            curr_cpu_ctl_env = guard_cpu_ctl_env;
        }
    }
};
#else
struct cpu_ctl_env_helper {
    void set_env( __TBB_CONTEXT_ARG1(task_group_context *) ) {}
    void restore_default() {}
};
#endif /* __TBB_FP_CONTEXT */

} // namespace internal
} // namespace tbb

#endif /* _TBB_scheduler_H */
