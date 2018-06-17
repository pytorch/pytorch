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

#ifndef _TBB_market_H
#define _TBB_market_H

#include "tbb/tbb_stddef.h"

#include "scheduler_common.h"
#include "tbb/atomic.h"
#include "tbb/spin_rw_mutex.h"
#include "../rml/include/rml_tbb.h"

#include "intrusive_list.h"

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
    #pragma warning (disable: 4244)
#endif

namespace tbb {

class task_group_context;

namespace internal {

//------------------------------------------------------------------------
// Class market
//------------------------------------------------------------------------

class market : no_copy, rml::tbb_client {
    friend class generic_scheduler;
    friend class arena;
    friend class tbb::interface7::internal::task_arena_base;
    template<typename SchedulerTraits> friend class custom_scheduler;
    friend class tbb::task_group_context;
private:
    friend void ITT_DoUnsafeOneTimeInitialization ();

    typedef intrusive_list<arena> arena_list_type;
    typedef intrusive_list<generic_scheduler> scheduler_list_type;

    //! Currently active global market
    static market* theMarket;

    typedef scheduler_mutex_type global_market_mutex_type;

    //! Mutex guarding creation/destruction of theMarket, insertions/deletions in my_arenas, and cancellation propagation
    static global_market_mutex_type  theMarketMutex;

    //! Lightweight mutex guarding accounting operations with arenas list
    typedef spin_rw_mutex arenas_list_mutex_type;
    arenas_list_mutex_type my_arenas_list_mutex;

    //! Pointer to the RML server object that services this TBB instance.
    rml::tbb_server* my_server;

    //! Maximal number of workers allowed for use by the underlying resource manager
    /** It can't be changed after market creation. **/
    unsigned my_num_workers_hard_limit;

    //! Current application-imposed limit on the number of workers (see set_active_num_workers())
    /** It can't be more than my_num_workers_hard_limit. **/
    unsigned my_num_workers_soft_limit;

    //! Number of workers currently requested from RML
    int my_num_workers_requested;

    //! First unused index of worker
    /** Used to assign indices to the new workers coming from RML, and busy part
        of my_workers array. **/
    atomic<unsigned> my_first_unused_worker_idx;

    //! Number of workers that were requested by all arenas
    int my_total_demand;

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    //! How many times mandatory concurrency was requested from the market
    int my_mandatory_num_requested;
#endif

#if __TBB_TASK_PRIORITY
    //! Highest priority among active arenas in the market.
    /** Arena priority level is its tasks highest priority (specified by arena's
        my_top_priority member).
        Arena is active when it has outstanding request for workers. Note that
        inactive arena may have workers lingering there for some time. **/
    intptr_t my_global_top_priority;

    //! Lowest priority among active arenas in the market.
    /** See also my_global_top_priority **/
    intptr_t my_global_bottom_priority;

    //! Tracks events that may bring tasks in offload areas to the top priority level.
    /** Incremented when global top priority is decremented or a task group priority
        is elevated to the current top level. **/
    uintptr_t my_global_reload_epoch;

    //! Information about arenas at a particular priority level
    struct priority_level_info {
        //! List of arenas at this priority level
        arena_list_type arenas;

        //! The first arena to be checked when idle worker seeks for an arena to enter
        /** The check happens in round-robin fashion. **/
        arena *next_arena;

        //! Total amount of workers requested by arenas at this priority level.
        int workers_requested;

        //! Maximal amount of workers the market can tell off to this priority level.
        int workers_available;
    }; // struct priority_level_info

    //! Information about arenas at different priority levels
    priority_level_info my_priority_levels[num_priority_levels];

#else /* !__TBB_TASK_PRIORITY */

    //! List of registered arenas
    arena_list_type my_arenas;

    //! The first arena to be checked when idle worker seeks for an arena to enter
    /** The check happens in round-robin fashion. **/
    arena *my_next_arena;
#endif /* !__TBB_TASK_PRIORITY */

    //! ABA prevention marker to assign to newly created arenas
    uintptr_t my_arenas_aba_epoch;

    //! Reference count controlling market object lifetime
    unsigned my_ref_count;

    //! Count of master threads attached
    unsigned my_public_ref_count;

    //! Stack size of worker threads
    size_t my_stack_size;

    //! Shutdown mode
    bool my_join_workers;

    //! The value indicating that the soft limit warning is unnecessary
    static const unsigned skip_soft_limit_warning = ~0U;

    //! Either workers soft limit to be reported via runtime_warning() or skip_soft_limit_warning
    unsigned my_workers_soft_limit_to_report;
#if __TBB_COUNT_TASK_NODES
    //! Net number of nodes that have been allocated from heap.
    /** Updated each time a scheduler or arena is destroyed. */
    atomic<intptr_t> my_task_node_count;
#endif /* __TBB_COUNT_TASK_NODES */

    //! Constructor
    market ( unsigned workers_soft_limit, unsigned workers_hard_limit, size_t stack_size );

    //! Factory method creating new market object
    static market& global_market ( bool is_public, unsigned max_num_workers = 0, size_t stack_size = 0 );

    //! Destroys and deallocates market object created by market::create()
    void destroy ();

#if __TBB_TASK_PRIORITY
    //! Returns next arena that needs more workers, or NULL.
    arena* arena_in_need ( arena* prev_arena );

    //! Recalculates the number of workers assigned to each arena at and below the specified priority.
    /** The actual number of workers servicing a particular arena may temporarily
        deviate from the calculated value. **/
    void update_allotment ( intptr_t highest_affected_priority );

    //! Changes arena's top priority and updates affected priority levels info in the market.
    void update_arena_top_priority ( arena& a, intptr_t newPriority );

    //! Changes market's global top priority and related settings.
    inline void update_global_top_priority ( intptr_t newPriority );

    //! Resets empty market's global top and bottom priority to the normal level.
    inline void reset_global_priority ();

    inline void advance_global_reload_epoch () {
        __TBB_store_with_release( my_global_reload_epoch, my_global_reload_epoch + 1 );
    }

    void assert_market_valid () const {
        __TBB_ASSERT( (my_priority_levels[my_global_top_priority].workers_requested > 0
                           && !my_priority_levels[my_global_top_priority].arenas.empty())
                       || (my_global_top_priority == my_global_bottom_priority &&
                           my_global_top_priority == normalized_normal_priority), NULL );
    }

#else /* !__TBB_TASK_PRIORITY */

    //! Recalculates the number of workers assigned to each arena in the list.
    /** The actual number of workers servicing a particular arena may temporarily
        deviate from the calculated value. **/
    void update_allotment () {
        if ( my_total_demand )
            update_allotment( my_arenas, my_total_demand, (int)my_num_workers_soft_limit );
    }

    //! Returns next arena that needs more workers, or NULL.
    arena* arena_in_need (arena*) {
        if(__TBB_load_with_acquire(my_total_demand) <= 0)
            return NULL;
        arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex, /*is_writer=*/false);
        return arena_in_need(my_arenas, my_next_arena);
    }
    void assert_market_valid () const {}
#endif /* !__TBB_TASK_PRIORITY */

    ////////////////////////////////////////////////////////////////////////////////
    // Helpers to unify code branches dependent on priority feature presence

    void insert_arena_into_list ( arena& a );

    void remove_arena_from_list ( arena& a );

    arena* arena_in_need ( arena_list_type &arenas, arena *&next );

    static int update_allotment ( arena_list_type& arenas, int total_demand, int max_workers );


    ////////////////////////////////////////////////////////////////////////////////
    // Implementation of rml::tbb_client interface methods

    version_type version () const __TBB_override { return 0; }

    unsigned max_job_count () const __TBB_override { return my_num_workers_hard_limit; }

    size_t min_stack_size () const __TBB_override { return worker_stack_size(); }

    policy_type policy () const __TBB_override { return throughput; }

    job* create_one_job () __TBB_override;

    void cleanup( job& j ) __TBB_override;

    void acknowledge_close_connection () __TBB_override;

    void process( job& j ) __TBB_override;

public:
    //! Creates an arena object
    /** If necessary, also creates global market instance, and boosts its ref count.
        Each call to create_arena() must be matched by the call to arena::free_arena(). **/
    static arena* create_arena ( int num_slots, int num_reserved_slots, size_t stack_size );

    //! Removes the arena from the market's list
    void try_destroy_arena ( arena*, uintptr_t aba_epoch );

    //! Removes the arena from the market's list
    void detach_arena ( arena& );

    //! Decrements market's refcount and destroys it in the end
    bool release ( bool is_public, bool blocking_terminate );

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    //! Imlpementation of mandatory concurrency enabling
    bool mandatory_concurrency_enable_impl ( arena *a, bool *enabled = NULL );

    //! Inform the master that there is an arena with mandatory concurrency
    bool mandatory_concurrency_enable ( arena *a );

    //! Inform the master that the arena is no more interested in mandatory concurrency
    void mandatory_concurrency_disable ( arena *a );
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */

    //! Request that arena's need in workers should be adjusted.
    /** Concurrent invocations are possible only on behalf of different arenas. **/
    void adjust_demand ( arena&, int delta );

    //! Used when RML asks for join mode during workers termination.
    bool must_join_workers () const { return my_join_workers; }

    //! Returns the requested stack size of worker threads.
    size_t worker_stack_size () const { return my_stack_size; }

    //! Set number of active workers
    static void set_active_num_workers( unsigned w );

    //! Reports active parallelism level according to user's settings
    static unsigned app_parallelism_limit();

#if _WIN32||_WIN64
    //! register master with the resource manager
    void register_master( ::rml::server::execution_resource_t& rsc_handle ) {
        __TBB_ASSERT( my_server, "RML server not defined?" );
        // the server may ignore registration and set master_exec_resource to NULL.
        my_server->register_master( rsc_handle );
    }

    //! unregister master with the resource manager
    void unregister_master( ::rml::server::execution_resource_t& rsc_handle ) const {
        my_server->unregister_master( rsc_handle );
    }
#endif /* WIN */

#if __TBB_TASK_GROUP_CONTEXT
    //! Finds all contexts affected by the state change and propagates the new state to them.
    /** The propagation is relayed to the market because tasks created by one
        master thread can be passed to and executed by other masters. This means
        that context trees can span several arenas at once and thus state change
        propagation cannot be generally localized to one arena only. **/
    template <typename T>
    bool propagate_task_group_state ( T task_group_context::*mptr_state, task_group_context& src, T new_state );
#endif /* __TBB_TASK_GROUP_CONTEXT */

#if __TBB_TASK_PRIORITY
    //! Lowers arena's priority is not higher than newPriority
    /** Returns true if arena priority was actually elevated. **/
    bool lower_arena_priority ( arena& a, intptr_t new_priority, uintptr_t old_reload_epoch );

    //! Makes sure arena's priority is not lower than newPriority
    /** Returns true if arena priority was elevated. Also updates arena's bottom
        priority boundary if necessary.

        This method is called whenever a user changes priority, because whether
        it was hiked or sunk can be determined for sure only under the lock used
        by this function. **/
    bool update_arena_priority ( arena& a, intptr_t new_priority );
#endif /* __TBB_TASK_PRIORITY */

#if __TBB_COUNT_TASK_NODES
    //! Net number of nodes that have been allocated from heap.
    /** Updated each time a scheduler or arena is destroyed. */
    void update_task_node_count( intptr_t delta ) { my_task_node_count += delta; }
#endif /* __TBB_COUNT_TASK_NODES */

#if __TBB_TASK_GROUP_CONTEXT
    //! List of registered master threads
    scheduler_list_type my_masters;

    //! Array of pointers to the registered workers
    /** Used by cancellation propagation mechanism.
        Must be the last data member of the class market. **/
    generic_scheduler* my_workers[1];
#endif /* __TBB_TASK_GROUP_CONTEXT */

    static unsigned max_num_workers() {
        global_market_mutex_type::scoped_lock lock( theMarketMutex );
        return theMarket? theMarket->my_num_workers_hard_limit : 0;
    }
}; // class market

} // namespace internal
} // namespace tbb

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (pop)
#endif // warning 4244 is back

#endif /* _TBB_market_H */
