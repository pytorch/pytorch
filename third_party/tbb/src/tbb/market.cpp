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
#include "tbb/global_control.h" // global_control::active_value

#include "market.h"
#include "tbb_main.h"
#include "governor.h"
#include "scheduler.h"
#include "itt_notify.h"

namespace tbb {
namespace internal {

void market::insert_arena_into_list ( arena& a ) {
#if __TBB_TASK_PRIORITY
    arena_list_type &arenas = my_priority_levels[a.my_top_priority].arenas;
    arena *&next = my_priority_levels[a.my_top_priority].next_arena;
#else /* !__TBB_TASK_PRIORITY */
    arena_list_type &arenas = my_arenas;
    arena *&next = my_next_arena;
#endif /* !__TBB_TASK_PRIORITY */
    arenas.push_front( a );
    if ( arenas.size() == 1 )
        next = &*arenas.begin();
}

void market::remove_arena_from_list ( arena& a ) {
#if __TBB_TASK_PRIORITY
    arena_list_type &arenas = my_priority_levels[a.my_top_priority].arenas;
    arena *&next = my_priority_levels[a.my_top_priority].next_arena;
#else /* !__TBB_TASK_PRIORITY */
    arena_list_type &arenas = my_arenas;
    arena *&next = my_next_arena;
#endif /* !__TBB_TASK_PRIORITY */
    arena_list_type::iterator it = next;
    __TBB_ASSERT( it != arenas.end(), NULL );
    if ( next == &a ) {
        if ( ++it == arenas.end() && arenas.size() > 1 )
            it = arenas.begin();
        next = &*it;
    }
    arenas.remove( a );
}

//------------------------------------------------------------------------
// market
//------------------------------------------------------------------------

market::market ( unsigned workers_soft_limit, unsigned workers_hard_limit, size_t stack_size )
    : my_num_workers_hard_limit(workers_hard_limit)
    , my_num_workers_soft_limit(workers_soft_limit)
#if __TBB_TASK_PRIORITY
    , my_global_top_priority(normalized_normal_priority)
    , my_global_bottom_priority(normalized_normal_priority)
#endif /* __TBB_TASK_PRIORITY */
    , my_ref_count(1)
    , my_stack_size(stack_size)
    , my_workers_soft_limit_to_report(workers_soft_limit)
{
#if __TBB_TASK_PRIORITY
    __TBB_ASSERT( my_global_reload_epoch == 0, NULL );
    my_priority_levels[normalized_normal_priority].workers_available = my_num_workers_soft_limit;
#endif /* __TBB_TASK_PRIORITY */

    // Once created RML server will start initializing workers that will need
    // global market instance to get worker stack size
    my_server = governor::create_rml_server( *this );
    __TBB_ASSERT( my_server, "Failed to create RML server" );
}

static unsigned calc_workers_soft_limit(unsigned workers_soft_limit, unsigned workers_hard_limit) {
    if( int soft_limit = market::app_parallelism_limit() )
        workers_soft_limit = soft_limit-1;
    else // if user set no limits (yet), use market's parameter
        workers_soft_limit = max( governor::default_num_threads() - 1, workers_soft_limit );
    if( workers_soft_limit >= workers_hard_limit )
        workers_soft_limit = workers_hard_limit-1;
    return workers_soft_limit;
}

market& market::global_market ( bool is_public, unsigned workers_requested, size_t stack_size ) {
    global_market_mutex_type::scoped_lock lock( theMarketMutex );
    market *m = theMarket;
    if( m ) {
        ++m->my_ref_count;
        const unsigned old_public_count = is_public? m->my_public_ref_count++ : /*any non-zero value*/1;
        lock.release();
        if( old_public_count==0 )
            set_active_num_workers( calc_workers_soft_limit(workers_requested, m->my_num_workers_hard_limit) );

        // do not warn if default number of workers is requested
        if( workers_requested != governor::default_num_threads()-1 ) {
            __TBB_ASSERT( skip_soft_limit_warning > workers_requested,
                          "skip_soft_limit_warning must be larger than any valid workers_requested" );
            unsigned soft_limit_to_report = m->my_workers_soft_limit_to_report;
            if( soft_limit_to_report < workers_requested ) {
                runtime_warning( "The number of workers is currently limited to %u. "
                                 "The request for %u workers is ignored. Further requests for more workers "
                                 "will be silently ignored until the limit changes.\n",
                                 soft_limit_to_report, workers_requested );
                // The race is possible when multiple threads report warnings.
                // We are OK with that, as there are just multiple warnings.
                internal::as_atomic(m->my_workers_soft_limit_to_report).
                    compare_and_swap(skip_soft_limit_warning, soft_limit_to_report);
            }

        }
        if( m->my_stack_size < stack_size )
            runtime_warning( "Thread stack size has been already set to %u. "
                             "The request for larger stack (%u) cannot be satisfied.\n",
                              m->my_stack_size, stack_size );
    }
    else {
        // TODO: A lot is done under theMarketMutex locked. Can anything be moved out?
        if( stack_size == 0 )
            stack_size = global_control::active_value(global_control::thread_stack_size);
        // Expecting that 4P is suitable for most applications.
        // Limit to 2P for large thread number.
        // TODO: ask RML for max concurrency and possibly correct hard_limit
        const unsigned factor = governor::default_num_threads()<=128? 4 : 2;
        // The requested number of threads is intentionally not considered in
        // computation of the hard limit, in order to separate responsibilities
        // and avoid complicated interactions between global_control and task_scheduler_init.
        // The market guarantees that at least 256 threads might be created.
        const unsigned workers_hard_limit = max(max(factor*governor::default_num_threads(), 256u), app_parallelism_limit());
        const unsigned workers_soft_limit = calc_workers_soft_limit(workers_requested, workers_hard_limit);
        // Create the global market instance
        size_t size = sizeof(market);
#if __TBB_TASK_GROUP_CONTEXT
        __TBB_ASSERT( __TBB_offsetof(market, my_workers) + sizeof(generic_scheduler*) == sizeof(market),
                      "my_workers must be the last data field of the market class");
        size += sizeof(generic_scheduler*) * (workers_hard_limit - 1);
#endif /* __TBB_TASK_GROUP_CONTEXT */
        __TBB_InitOnce::add_ref();
        void* storage = NFS_Allocate(1, size, NULL);
        memset( storage, 0, size );
        // Initialize and publish global market
        m = new (storage) market( workers_soft_limit, workers_hard_limit, stack_size );
        if( is_public )
            m->my_public_ref_count = 1;
        theMarket = m;
        // This check relies on the fact that for shared RML default_concurrency==max_concurrency
        if ( !governor::UsePrivateRML && m->my_server->default_concurrency() < workers_soft_limit )
            runtime_warning( "RML might limit the number of workers to %u while %u is requested.\n"
                    , m->my_server->default_concurrency(), workers_soft_limit );
    }
    return *m;
}

void market::destroy () {
#if __TBB_COUNT_TASK_NODES
    if ( my_task_node_count )
        runtime_warning( "Leaked %ld task objects\n", (long)my_task_node_count );
#endif /* __TBB_COUNT_TASK_NODES */
    this->market::~market(); // qualified to suppress warning
    NFS_Free( this );
    __TBB_InitOnce::remove_ref();
}

bool market::release ( bool is_public, bool blocking_terminate ) {
    __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
    bool do_release = false;
    {
        global_market_mutex_type::scoped_lock lock( theMarketMutex );
        if ( blocking_terminate ) {
            __TBB_ASSERT( is_public, "Only an object with a public reference can request the blocking terminate" );
            while ( my_public_ref_count == 1 && my_ref_count > 1 ) {
                lock.release();
                // To guarantee that request_close_connection() is called by the last master, we need to wait till all
                // references are released. Re-read my_public_ref_count to limit waiting if new masters are created.
                // Theoretically, new private references to the market can be added during waiting making it potentially
                // endless.
                // TODO: revise why the weak scheduler needs market's pointer and try to remove this wait.
                // Note that the market should know about its schedulers for cancelation/exception/priority propagation,
                // see e.g. task_group_context::cancel_group_execution()
                while ( __TBB_load_with_acquire( my_public_ref_count ) == 1 && __TBB_load_with_acquire( my_ref_count ) > 1 )
                    __TBB_Yield();
                lock.acquire( theMarketMutex );
            }
        }
        if ( is_public ) {
            __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
            __TBB_ASSERT( my_public_ref_count, NULL );
            --my_public_ref_count;
        }
        if ( --my_ref_count == 0 ) {
            __TBB_ASSERT( !my_public_ref_count, NULL );
            do_release = true;
            theMarket = NULL;
        }
    }
    if( do_release ) {
        __TBB_ASSERT( !__TBB_load_with_acquire(my_public_ref_count), "No public references remain if we remove the market." );
        // inform RML that blocking termination is required
        my_join_workers = blocking_terminate;
        my_server->request_close_connection();
        return blocking_terminate;
    }
    return false;
}

void market::set_active_num_workers ( unsigned soft_limit ) {
    int old_requested=0, requested=0;
    bool need_mandatory = false;
    market *m;

    {
        global_market_mutex_type::scoped_lock lock( theMarketMutex );
        if ( !theMarket )
            return; // actual value will be used at market creation
        m = theMarket;
        ++m->my_ref_count;
    }
    // have my_ref_count for market, use it safely
    {
        arenas_list_mutex_type::scoped_lock lock( m->my_arenas_list_mutex );
        __TBB_ASSERT(soft_limit <= m->my_num_workers_hard_limit, NULL);
        m->my_num_workers_soft_limit = soft_limit;
        // report only once after new soft limit value is set
        m->my_workers_soft_limit_to_report = soft_limit;

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        // updates soft_limit to zero must be postponed
        // while mandatory parallelism is enabled
        if( !(m->my_mandatory_num_requested && !soft_limit) )
#endif
        {
            const int demand =
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
                m->my_mandatory_num_requested? 0 :
#endif
                m->my_total_demand;
            requested = min(demand, (int)soft_limit);
            old_requested = m->my_num_workers_requested;
            m->my_num_workers_requested = requested;
#if __TBB_TASK_PRIORITY
            m->my_priority_levels[m->my_global_top_priority].workers_available = soft_limit;
            m->update_allotment( m->my_global_top_priority );
#else
            m->update_allotment();
#endif
        }
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        if( !m->my_mandatory_num_requested && !soft_limit ) {
            // enable mandatory concurrency, if enqueued tasks are found
            // and zero soft_limit requested
#if __TBB_TASK_PRIORITY
            for( int p = m->my_global_top_priority; p >= m->my_global_bottom_priority; --p ) {
                priority_level_info &pl = m->my_priority_levels[p];
                arena_list_type &arenas = pl.arenas;
#else
                const int p = 0;
                arena_list_type &arenas = m->my_arenas;
#endif /* __TBB_TASK_PRIORITY */
                for( arena_list_type::iterator it = arenas.begin(); it != arenas.end(); ++it ) {
                    if( !it->my_task_stream.empty(p) ) {
                        // switch local_mandatory to global_mandatory unconditionally
                        if( m->mandatory_concurrency_enable_impl( &*it ) )
                            need_mandatory = true;
                    }
                }
#if __TBB_TASK_PRIORITY
            }
#endif /* __TBB_TASK_PRIORITY */
        }
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */
    }
    // adjust_job_count_estimate must be called outside of any locks
    int delta = requested - old_requested;
    if( need_mandatory ) ++delta;
    if( delta!=0 )
        m->my_server->adjust_job_count_estimate( delta );
    // release internal market reference to match ++m->my_ref_count above
    m->release( /*is_public=*/false, /*blocking_terminate=*/false );
}

bool governor::does_client_join_workers (const tbb::internal::rml::tbb_client &client) {
    return ((const market&)client).must_join_workers();
}

arena* market::create_arena ( int num_slots, int num_reserved_slots, size_t stack_size ) {
    __TBB_ASSERT( num_slots > 0, NULL );
    __TBB_ASSERT( num_reserved_slots <= num_slots, NULL );
    // Add public market reference for master thread/task_arena (that adds an internal reference in exchange).
    market &m = global_market( /*is_public=*/true, num_slots-num_reserved_slots, stack_size );

    arena& a = arena::allocate_arena( m, num_slots, num_reserved_slots );
    // Add newly created arena into the existing market's list.
    arenas_list_mutex_type::scoped_lock lock(m.my_arenas_list_mutex);
    m.insert_arena_into_list(a);
    return &a;
}

/** This method must be invoked under my_arenas_list_mutex. **/
void market::detach_arena ( arena& a ) {
    __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
    __TBB_ASSERT( !a.my_slots[0].my_scheduler, NULL );
    remove_arena_from_list(a);
    if ( a.my_aba_epoch == my_arenas_aba_epoch )
        ++my_arenas_aba_epoch;
}

void market::try_destroy_arena ( arena* a, uintptr_t aba_epoch ) {
    bool locked = true;
    __TBB_ASSERT( a, NULL );
    // we hold reference to the market, so it cannot be destroyed at any moment here
    __TBB_ASSERT( this == theMarket, NULL );
    __TBB_ASSERT( my_ref_count!=0, NULL );
    my_arenas_list_mutex.lock();
    assert_market_valid();
#if __TBB_TASK_PRIORITY
    // scan all priority levels, not only in [my_global_bottom_priority;my_global_top_priority]
    // range, because arena to be destroyed can have no outstanding request for workers
    for ( int p = num_priority_levels-1; p >= 0; --p ) {
        priority_level_info &pl = my_priority_levels[p];
        arena_list_type &my_arenas = pl.arenas;
#endif /* __TBB_TASK_PRIORITY */
        arena_list_type::iterator it = my_arenas.begin();
        for ( ; it != my_arenas.end(); ++it ) {
            if ( a == &*it ) {
                if ( it->my_aba_epoch == aba_epoch ) {
                    // Arena is alive
                    if ( !a->my_num_workers_requested && !a->my_references ) {
                        __TBB_ASSERT( !a->my_num_workers_allotted && (a->my_pool_state == arena::SNAPSHOT_EMPTY || !a->my_max_num_workers), "Inconsistent arena state" );
                        // Arena is abandoned. Destroy it.
                        detach_arena( *a );
                        my_arenas_list_mutex.unlock();
                        locked = false;
                        a->free_arena();
                    }
                }
                if (locked)
                    my_arenas_list_mutex.unlock();
                return;
            }
        }
#if __TBB_TASK_PRIORITY
    }
#endif /* __TBB_TASK_PRIORITY */
    my_arenas_list_mutex.unlock();
}

/** This method must be invoked under my_arenas_list_mutex. **/
arena* market::arena_in_need ( arena_list_type &arenas, arena *&next ) {
    if ( arenas.empty() )
        return NULL;
    arena_list_type::iterator it = next;
    __TBB_ASSERT( it != arenas.end(), NULL );
    do {
        arena& a = *it;
        if ( ++it == arenas.end() )
            it = arenas.begin();
        if( a.num_workers_active() < a.my_num_workers_allotted
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
            && !a.recall_by_mandatory_request()
#endif
            ) {
            a.my_references += arena::ref_worker;
            as_atomic(next) = &*it; // a subject for innocent data race under the reader lock
            // TODO: rework global round robin policy to local or random to avoid this write
            return &a;
        }
    } while ( it != next );
    return NULL;
}

int market::update_allotment ( arena_list_type& arenas, int workers_demand, int max_workers ) {
    __TBB_ASSERT( workers_demand, NULL );
    max_workers = min(workers_demand, max_workers);
    int carry = 0;
    int assigned = 0;
    arena_list_type::iterator it = arenas.begin();
    for ( ; it != arenas.end(); ++it ) {
        arena& a = *it;
        if ( a.my_num_workers_requested <= 0 ) {
            __TBB_ASSERT( !a.my_num_workers_allotted, NULL );
            continue;
        }
        int tmp = a.my_num_workers_requested * max_workers + carry;
        int allotted = tmp / workers_demand;
        carry = tmp % workers_demand;
        // a.my_num_workers_requested may temporarily exceed a.my_max_num_workers
        allotted = min( allotted, (int)a.my_max_num_workers );
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        if ( !allotted && a.must_have_concurrency() )
            allotted = 1;
#endif
        a.my_num_workers_allotted = allotted;
        assigned += allotted;
    }
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    __TBB_ASSERT( assigned <= workers_demand, NULL ); // weaker assertion due to enforced allotment
#else
    __TBB_ASSERT( assigned <= max_workers, NULL );
#endif
    return assigned;
}

#if __TBB_TASK_PRIORITY
inline void market::update_global_top_priority ( intptr_t newPriority ) {
    GATHER_STATISTIC( ++governor::local_scheduler_if_initialized()->my_counters.market_prio_switches );
    my_global_top_priority = newPriority;
    my_priority_levels[newPriority].workers_available =
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        my_mandatory_num_requested && !my_num_workers_soft_limit ? 1 :
#endif
        my_num_workers_soft_limit;
    advance_global_reload_epoch();
}

inline void market::reset_global_priority () {
    my_global_bottom_priority = normalized_normal_priority;
    update_global_top_priority(normalized_normal_priority);
}

arena* market::arena_in_need ( arena* prev_arena )
{
    suppress_unused_warning(prev_arena);
    if( as_atomic(my_total_demand) <= 0 )
        return NULL;
    arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex, /*is_writer=*/false);
    assert_market_valid();
    int p = my_global_top_priority;
    arena *a = NULL;
    do {
        priority_level_info &pl = my_priority_levels[p];
        a = arena_in_need( pl.arenas, pl.next_arena );
        // TODO: When refactoring task priority code, take into consideration the
        // __TBB_TRACK_PRIORITY_LEVEL_SATURATION sections from earlier versions of TBB
    } while ( !a && --p >= my_global_bottom_priority );
    return a;
}

void market::update_allotment ( intptr_t highest_affected_priority ) {
    intptr_t i = highest_affected_priority;
    int available = my_priority_levels[i].workers_available;
    for ( ; i >= my_global_bottom_priority; --i ) {
        priority_level_info &pl = my_priority_levels[i];
        pl.workers_available = available;
        if ( pl.workers_requested ) {
            available -= update_allotment( pl.arenas, pl.workers_requested, available );
            if ( available < 0 ) { // TODO: assertion?
                available = 0;
                break;
            }
        }
    }
    __TBB_ASSERT( i <= my_global_bottom_priority || !available, NULL );
    for ( --i; i >= my_global_bottom_priority; --i ) {
        priority_level_info &pl = my_priority_levels[i];
        pl.workers_available = 0;
        arena_list_type::iterator it = pl.arenas.begin();
        for ( ; it != pl.arenas.end(); ++it ) {
            __TBB_ASSERT( it->my_num_workers_requested >= 0 || !it->my_num_workers_allotted, NULL );
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
            it->my_num_workers_allotted = it->must_have_concurrency() ? 1 : 0;
#else
            it->my_num_workers_allotted = 0;
#endif
        }
    }
}
#endif /* __TBB_TASK_PRIORITY */

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
bool market::mandatory_concurrency_enable_impl ( arena *a, bool *enabled ) {
    if( a->my_concurrency_mode==arena_base::cm_enforced_global ) {
        if( enabled )
            *enabled = false;
        return false;
    }
    if( enabled )
        *enabled = true;
    a->my_max_num_workers = 1;
    a->my_concurrency_mode = arena_base::cm_enforced_global;
#if __TBB_TASK_PRIORITY
    priority_level_info &pl = my_priority_levels[a->my_top_priority];
    pl.workers_requested++;
    if( my_global_top_priority < a->my_top_priority ) {
        my_global_top_priority = a->my_top_priority;
        advance_global_reload_epoch();
    }
#endif
    a->my_num_workers_requested++;
    a->my_num_workers_allotted++;
    if( 1 == ++my_mandatory_num_requested ) {
        my_total_demand++;
        return true;
    }
    return false;
}

bool market::mandatory_concurrency_enable ( arena *a ) {
    bool add_thread;
    bool enabled;
    {
        arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);
        add_thread = mandatory_concurrency_enable_impl(a, &enabled);
    }
    if( add_thread )
        my_server->adjust_job_count_estimate( 1 );
    return enabled;
}

void market::mandatory_concurrency_disable ( arena *a ) {
    bool remove_thread = false;
    int delta_adjust_demand = 0;

    {
        arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);

        if( a->my_concurrency_mode!=arena_base::cm_enforced_global  )
            return;
        __TBB_ASSERT( a->my_max_num_workers==1, NULL );
        a->my_max_num_workers = 0;
#if __TBB_TASK_PRIORITY
        if ( a->my_top_priority != normalized_normal_priority ) {
            update_arena_top_priority( *a, normalized_normal_priority );
        }
        a->my_bottom_priority = normalized_normal_priority;
#endif

        int val = --my_mandatory_num_requested;
        __TBB_ASSERT_EX( val >= 0, NULL );
        if( val == 0 ) {
            my_total_demand--;
            remove_thread = true;
        }
        a->my_num_workers_requested--;
        if (a->my_num_workers_requested > 0)
            delta_adjust_demand = a->my_num_workers_requested;
        else
            a->my_num_workers_allotted = 0;

#if __TBB_TASK_PRIORITY
        priority_level_info &pl = my_priority_levels[a->my_top_priority];
        pl.workers_requested--;
        intptr_t p = my_global_top_priority;
        for (; !my_priority_levels[p].workers_requested && p>0; p--)
            ;
        if( !p )
            reset_global_priority();
        else if( p!= my_global_top_priority )
            update_global_top_priority(p);
#endif
        a->my_concurrency_mode = arena::cm_normal;
    }
    if( delta_adjust_demand )
        adjust_demand( *a, -delta_adjust_demand );
    if( remove_thread )
        my_server->adjust_job_count_estimate( -1 );
}
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */

void market::adjust_demand ( arena& a, int delta ) {
    __TBB_ASSERT( theMarket, "market instance was destroyed prematurely?" );
    if ( !delta )
        return;
    my_arenas_list_mutex.lock();
    int prev_req = a.my_num_workers_requested;
    a.my_num_workers_requested += delta;
    if ( a.my_num_workers_requested <= 0 ) {
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        // must not recall worker from arena with mandatory parallelism
        if ( a.my_market->my_mandatory_num_requested && a.my_concurrency_mode!=arena_base::cm_normal )
            a.my_num_workers_allotted = 1;
        else
#endif
            a.my_num_workers_allotted = 0;
        if ( prev_req <= 0 ) {
            my_arenas_list_mutex.unlock();
            return;
        }
        delta = -prev_req;
    }
    else if ( prev_req < 0 ) {
        delta = a.my_num_workers_requested;
    }
    my_total_demand += delta;
#if !__TBB_TASK_PRIORITY
    update_allotment();
#else /* !__TBB_TASK_PRIORITY */
    intptr_t p = a.my_top_priority;
    priority_level_info &pl = my_priority_levels[p];
    pl.workers_requested += delta;
    __TBB_ASSERT( pl.workers_requested >= 0, NULL );
    if ( a.my_num_workers_requested <= 0 ) {
        if ( a.my_top_priority != normalized_normal_priority ) {
            GATHER_STATISTIC( ++governor::local_scheduler_if_initialized()->my_counters.arena_prio_resets );
            update_arena_top_priority( a, normalized_normal_priority );
        }
        a.my_bottom_priority = normalized_normal_priority;
    }
    if ( p == my_global_top_priority ) {
        if ( !pl.workers_requested ) {
            while ( --p >= my_global_bottom_priority && !my_priority_levels[p].workers_requested )
                continue;
            if ( p < my_global_bottom_priority )
                reset_global_priority();
            else
                update_global_top_priority(p);
        }
        update_allotment( my_global_top_priority );
    }
    else if ( p > my_global_top_priority ) {
        __TBB_ASSERT( pl.workers_requested > 0, NULL );
        // TODO: investigate if the following invariant is always valid
        __TBB_ASSERT( a.my_num_workers_requested >= 0, NULL );
        update_global_top_priority(p);
        a.my_num_workers_allotted = min( (int)my_num_workers_soft_limit, a.my_num_workers_requested );
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        // must not recall worker from arena with mandatory parallelism
        if ( !a.my_num_workers_allotted && a.my_num_workers_requested
             && a.my_market->my_mandatory_num_requested && a.my_concurrency_mode!=arena_base::cm_normal )
            a.my_num_workers_allotted = 1;
#endif
        my_priority_levels[p - 1].workers_available = my_num_workers_soft_limit - a.my_num_workers_allotted;
        update_allotment( p - 1 );
    }
    else if ( p == my_global_bottom_priority ) {
        if ( !pl.workers_requested ) {
            while ( ++p <= my_global_top_priority && !my_priority_levels[p].workers_requested )
                continue;
            if ( p > my_global_top_priority )
                reset_global_priority();
            else
                my_global_bottom_priority = p;
        }
        else
            update_allotment( p );
    }
    else if ( p < my_global_bottom_priority ) {
        int prev_bottom = my_global_bottom_priority;
        my_global_bottom_priority = p;
        update_allotment( prev_bottom );
    }
    else {
        __TBB_ASSERT( my_global_bottom_priority < p && p < my_global_top_priority, NULL );
        update_allotment( p );
    }
    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority || a.my_num_workers_requested<=0, NULL );
    assert_market_valid();
#endif /* !__TBB_TASK_PRIORITY */
    if ( delta > 0 ) {
        // can't overflow soft_limit, but remember values request by arenas in
        // my_total_demand to not prematurely release workers to RML
        if ( my_num_workers_requested+delta > (int)my_num_workers_soft_limit )
            delta = my_num_workers_soft_limit - my_num_workers_requested;
    } else {
        // the number of workers should not be decreased below my_total_demand
        if ( my_num_workers_requested+delta < my_total_demand )
            delta = min(my_total_demand, (int)my_num_workers_soft_limit) - my_num_workers_requested;
    }
    my_num_workers_requested += delta;
    __TBB_ASSERT( my_num_workers_requested <= (int)my_num_workers_soft_limit, NULL );

    my_arenas_list_mutex.unlock();
    // Must be called outside of any locks
    my_server->adjust_job_count_estimate( delta );
    GATHER_STATISTIC( governor::local_scheduler_if_initialized() ? ++governor::local_scheduler_if_initialized()->my_counters.gate_switches : 0 );
}

void market::process( job& j ) {
    generic_scheduler& s = static_cast<generic_scheduler&>(j);
    arena *a = NULL;
    __TBB_ASSERT( governor::is_set(&s), NULL );
    enum {
        query_interval = 1000,
        first_interval = 1
    };
    for(int i = first_interval; ; i--) {
        while ( (a = arena_in_need(a)) )
        {
            a->process(s);
            i = first_interval;
        }
        // Workers leave market because there is no arena in need. It can happen earlier than
        // adjust_job_count_estimate() decreases my_slack and RML can put this thread to sleep.
        // It might result in a busy-loop checking for my_slack<0 and calling this method instantly.
        // first_interval>0 and the yield refines this spinning.
        if( i > 0 )
            __TBB_Yield();
        else
#if !__TBB_SLEEP_PERMISSION
            break;
#else
        { // i == 0
#if __TBB_TASK_PRIORITY
            arena_list_type &al = my_priority_levels[my_global_top_priority].arenas;
#else /* __TBB_TASK_PRIORITY */
            arena_list_type &al = my_arenas;
#endif /* __TBB_TASK_PRIORITY */
            if( al.empty() ) // races if any are innocent TODO: replace by an RML query interface
                break; // no arenas left, perhaps going to shut down
            if( the_global_observer_list.ask_permission_to_leave() )
                break; // go sleep
            __TBB_Yield();
            i = query_interval;
        }
#endif// !__TBB_SLEEP_PERMISSION
    }
    GATHER_STATISTIC( ++s.my_counters.market_roundtrips );
}

void market::cleanup( job& j ) {
    __TBB_ASSERT( theMarket != this, NULL );
    generic_scheduler& s = static_cast<generic_scheduler&>(j);
    generic_scheduler* mine = governor::local_scheduler_if_initialized();
    __TBB_ASSERT( !mine || mine->is_worker(), NULL );
    if( mine!=&s ) {
        governor::assume_scheduler( &s );
        generic_scheduler::cleanup_worker( &s, mine!=NULL );
        governor::assume_scheduler( mine );
    } else {
        generic_scheduler::cleanup_worker( &s, true );
    }
}

void market::acknowledge_close_connection() {
    destroy();
}

::rml::job* market::create_one_job() {
    unsigned index = ++my_first_unused_worker_idx;
    __TBB_ASSERT( index > 0, NULL );
    ITT_THREAD_SET_NAME(_T("TBB Worker Thread"));
    // index serves as a hint decreasing conflicts between workers when they migrate between arenas
    generic_scheduler* s = generic_scheduler::create_worker( *this, index );
#if __TBB_TASK_GROUP_CONTEXT
    __TBB_ASSERT( index <= my_num_workers_hard_limit, NULL );
    __TBB_ASSERT( !my_workers[index - 1], NULL );
    my_workers[index - 1] = s;
#endif /* __TBB_TASK_GROUP_CONTEXT */
    return s;
}

#if __TBB_TASK_PRIORITY
void market::update_arena_top_priority ( arena& a, intptr_t new_priority ) {
    GATHER_STATISTIC( ++governor::local_scheduler_if_initialized()->my_counters.arena_prio_switches );
    __TBB_ASSERT( a.my_top_priority != new_priority, NULL );
    priority_level_info &prev_level = my_priority_levels[a.my_top_priority],
                        &new_level = my_priority_levels[new_priority];
    remove_arena_from_list(a);
    a.my_top_priority = new_priority;
    insert_arena_into_list(a);
    as_atomic( a.my_reload_epoch ).fetch_and_increment<tbb::release>(); // TODO: synch with global reload epoch in order to optimize usage of local reload epoch
    prev_level.workers_requested -= a.my_num_workers_requested;
    new_level.workers_requested += a.my_num_workers_requested;
    __TBB_ASSERT( prev_level.workers_requested >= 0 && new_level.workers_requested >= 0, NULL );
}

bool market::lower_arena_priority ( arena& a, intptr_t new_priority, uintptr_t old_reload_epoch ) {
    // TODO: replace the lock with a try_lock loop which performs a double check of the epoch
    arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);
    if ( a.my_reload_epoch != old_reload_epoch ) {
        assert_market_valid();
        return false;
    }
    __TBB_ASSERT( a.my_top_priority > new_priority, NULL );
    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority, NULL );

    intptr_t p = a.my_top_priority;
    update_arena_top_priority( a, new_priority );
    if ( a.my_num_workers_requested > 0 ) {
        if ( my_global_bottom_priority > new_priority ) {
            my_global_bottom_priority = new_priority;
        }
        if ( p == my_global_top_priority && !my_priority_levels[p].workers_requested ) {
            // Global top level became empty
            for ( --p; p>my_global_bottom_priority && !my_priority_levels[p].workers_requested; --p ) continue;
            update_global_top_priority(p);
        }
        update_allotment( p );
    }

    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority, NULL );
    assert_market_valid();
    return true;
}

bool market::update_arena_priority ( arena& a, intptr_t new_priority ) {
    // TODO: do not acquire this global lock while checking arena's state.
    arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);

    tbb::internal::assert_priority_valid(new_priority);
    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority || a.my_num_workers_requested <= 0, NULL );
    assert_market_valid();
    if ( a.my_top_priority == new_priority ) {
        return false;
    }
    else if ( a.my_top_priority > new_priority ) {
        if ( a.my_bottom_priority > new_priority )
            a.my_bottom_priority = new_priority;
        return false;
    }
    else if ( a.my_num_workers_requested <= 0 ) {
        return false;
    }

    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority, NULL );

    intptr_t p = a.my_top_priority;
    intptr_t highest_affected_level = max(p, new_priority);
    update_arena_top_priority( a, new_priority );

    if ( my_global_top_priority < new_priority ) {
        update_global_top_priority(new_priority);
    }
    else if ( my_global_top_priority == new_priority ) {
        advance_global_reload_epoch();
    }
    else {
        __TBB_ASSERT( new_priority < my_global_top_priority, NULL );
        __TBB_ASSERT( new_priority > my_global_bottom_priority, NULL );
        if ( p == my_global_top_priority && !my_priority_levels[p].workers_requested ) {
            // Global top level became empty
            __TBB_ASSERT( my_global_bottom_priority < p, NULL );
            for ( --p; !my_priority_levels[p].workers_requested; --p ) continue;
            __TBB_ASSERT( p >= new_priority, NULL );
            update_global_top_priority(p);
            highest_affected_level = p;
        }
    }
    if ( p == my_global_bottom_priority ) {
        // Arena priority was increased from the global bottom level.
        __TBB_ASSERT( p < new_priority, NULL );
        __TBB_ASSERT( new_priority <= my_global_top_priority, NULL );
        while ( my_global_bottom_priority < my_global_top_priority
                && !my_priority_levels[my_global_bottom_priority].workers_requested )
            ++my_global_bottom_priority;
        __TBB_ASSERT( my_global_bottom_priority <= new_priority, NULL );
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        const bool enforced_concurrency = my_mandatory_num_requested && a.must_have_concurrency();
#else
        const bool enforced_concurrency = false;
#endif
        __TBB_ASSERT_EX( enforced_concurrency || my_priority_levels[my_global_bottom_priority].workers_requested > 0, NULL );
    }
    update_allotment( highest_affected_level );

    __TBB_ASSERT( my_global_top_priority >= a.my_top_priority, NULL );
    assert_market_valid();
    return true;
}
#endif /* __TBB_TASK_PRIORITY */

} // namespace internal
} // namespace tbb
