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

#ifndef _TBB_tbb_statistics_H
#define _TBB_tbb_statistics_H

/**
    This file defines parameters of the internal statistics collected by the TBB
    library (currently by the task scheduler only).

    Statistics is accumulated separately in each thread and is dumped when
    the scheduler instance associated with the given  thread is destroyed.
    For apps with multiple master threads or with the same master repeatedly
    initializing and then deinitializing task scheduler this results in TBB
    workers statistics getting inseparably mixed.

    Therefore statistics is accumulated in arena slots, and should be dumped
    when arena is destroyed. This separates statistics collected for each
    scheduler activity region in each master thread.

    With the current RML implementation (TBB 2.2, 3.0) to avoid complete loss of
    statistics data during app shutdown (because of lazy workers deinitialization
    logic) set __TBB_STATISTICS_EARLY_DUMP macro to write the statistics at the
    moment a master thread deinitializes its scheduler. This may happen a little
    earlier than the moment of arena destruction resulting in the following undesired
    (though usually tolerable) effects:
    - a few events related to unsuccessful stealing or thread pool activity may be lost,
    - statistics may be substantially incomplete in case of FIFO tasks used in
      the FAF mode.

    Macro __TBB_STATISTICS_STDOUT and global variable __TBB_ActiveStatisticsGroups
    defined below can be used to configure the statistics output.

    To add new counter:
    1) Insert it into the appropriate group range in statistics_counters;
    2) Insert the corresponding field title into StatFieldTitles (preserving
       relative order of the fields).

    To add new counters group:
    1) Insert new group bit flag into statistics_groups;
    2) Insert the new group title into StatGroupTitles (preserving
       relative order of the groups).
    3) Add counter belonging to the new group as described above
**/

#include "tbb/tbb_stddef.h"

#ifndef __TBB_STATISTICS
#define __TBB_STATISTICS 0
#endif /* __TBB_STATISTICS */

#if __TBB_STATISTICS

#include <string.h>  // for memset

//! Dump counters into stdout as well.
/** By default statistics counters are written to the file "statistics.txt" only. **/
#define __TBB_STATISTICS_STDOUT 1

//! Dump only totals for all threads in the given arena
/** By default statistics counters for each arena slot are dumped separately, as
    well as the subtotal for workers. **/
#define __TBB_STATISTICS_TOTALS_ONLY 1

//! Dump statistics for an arena when its master completes
/** By default (when this macro is not set) the statistics is sent to output when
    arena object is destroyed. But with the current lazy workers termination
    logic default behavior may result in losing all statistics output. **/
#define __TBB_STATISTICS_EARLY_DUMP 1

#define GATHER_STATISTIC(x) (x)

namespace tbb {
namespace internal {

//! Groups of statistics counters.
/** The order of enumerators must be the same as the order of the corresponding
    field groups in the statistics_counters structure. **/
enum statistics_groups {
    sg_task_allocation = 0x01,
    sg_task_execution = 0x02,
    sg_stealing = 0x04,
    sg_affinity = 0x08,
    sg_arena = 0x10,
    sg_market = 0x20,
    sg_prio = 0x40,
    sg_prio_ex = 0x80,
    // List end marker. Insert new groups only before it.
    sg_end
};

//! Groups of counters to output
const uintptr_t __TBB_ActiveStatisticsGroups = sg_task_execution | sg_stealing | sg_affinity | sg_arena | sg_market;

//! A set of various statistics counters that are updated by the library on per thread basis.
/** All the fields must be of the same type (statistics_counters::counter_type).
    This is necessary to allow reinterpreting this structure as an array. **/
struct statistics_counters {
    typedef long counter_type;

    // Group: sg_task_allocation
    // Counters in this group can have negative values as the tasks migrate across
    // threads while the associated counters are updated in the current thread only
    // to avoid data races

    //! Number of tasks allocated and not yet destroyed
    counter_type active_tasks;
    //! Number of task corpses stored for future reuse
    counter_type free_list_length;
    //! Number of big tasks allocated during the run
    /** To find total number of tasks malloc'd, compute (big_tasks+my_small_task_count) */
    counter_type big_tasks;

    // Group: sg_task_execution

    //! Number of tasks executed
    counter_type tasks_executed;
    //! Number of elided spawns
    counter_type spawns_bypassed;

    // Group: sg_stealing

    //! Number of tasks successfully stolen
    counter_type steals_committed;
    //! Number of failed stealing attempts
    counter_type steals_failed;
    //! Number of failed attempts to lock victim's task pool
    counter_type thieves_conflicts;
    //! Number of times thief backed off because of the collision with the owner
    counter_type thief_backoffs;

    // Group: sg_affinity

    //! Number of tasks received from mailbox
    counter_type mails_received;
    //! Number of affinitized tasks executed by the owner
    /** Goes as "revoked" in statistics printout. **/
    counter_type proxies_executed;
    //! Number of affinitized tasks intercepted by thieves
    counter_type proxies_stolen;
    //! Number of proxy bypasses by thieves during stealing
    counter_type proxies_bypassed;
    //! Number of affinitized tasks executed by the owner via scheduler bypass mechanism
    counter_type affinity_ignored;

    // Group: sg_arena

    //! Number of times the state of arena switched between "full" and "empty"
    counter_type gate_switches;
    //! Number of times workers left an arena and returned into the market
    counter_type arena_roundtrips;
    // !Average concurrency level of this arena
    counter_type avg_arena_concurrency;
    //! Average assigned priority
    counter_type avg_assigned_workers;

    // Group: sg_market

    //! Number of times workers left the market and returned into RML
    counter_type market_roundtrips;

    // Group; sg_prio

    //! Number of arena priority switches
    counter_type arena_prio_switches;
    //! Number of market priority switches
    counter_type market_prio_switches;
    //! Number of arena priority switches
    counter_type arena_prio_resets;
    //! Number of reference priority source fixups to avoid deadlock
    counter_type prio_ref_fixups;
    //! Average arena priority
    counter_type avg_arena_prio;
    //! Average market priority
    counter_type avg_market_prio;

    // Group; sg_prio_ex

    //! Number of times local task pools were winnowed
    counter_type prio_winnowings;
    //! Number of times secondary task pools were searched for top priority tasks
    counter_type prio_reloads;
    //! Number of times secondary task pools were abandoned by quitting workers
    counter_type prio_orphanings;
    //! Number of tasks offloaded into secondary task pools
    counter_type prio_tasks_offloaded;
    //! Number of tasks reloaded from secondary task pools
    counter_type prio_tasks_reloaded;

    // Constructor and helpers

    statistics_counters() { reset(); }

    void reset () { memset( this, 0, sizeof(statistics_counters) ); }

    counter_type& field ( size_t index ) { return reinterpret_cast<counter_type*>(this)[index]; }

    const counter_type& field ( size_t index ) const { return reinterpret_cast<const counter_type*>(this)[index]; }

    static size_t size () { return sizeof(statistics_counters) / sizeof(counter_type); }

    const statistics_counters& operator += ( const statistics_counters& rhs ) {
        for ( size_t i = 0; i < size(); ++i )
            field(i) += rhs.field(i);
        return *this;
    }
}; // statistics_counters

static const size_t workers_counters_total = (size_t)-1;
static const size_t arena_counters_total = (size_t)-2;

void dump_statistics ( const statistics_counters& c, size_t id );

} // namespace internal
} // namespace tbb

#else /* !__TBB_STATISTICS */

#define GATHER_STATISTIC(x) ((void)0)

#endif /* !__TBB_STATISTICS */

#endif /* _TBB_tbb_statistics_H */
