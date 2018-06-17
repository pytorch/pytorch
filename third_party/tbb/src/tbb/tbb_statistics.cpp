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

#include "tbb_statistics.h"

#if __TBB_STATISTICS

#include <climits>
#include <cstdarg>
#if __TBB_STATISTICS_STDOUT
#include <cstdio>
#endif

#include "tbb/spin_mutex.h"

namespace tbb {
namespace internal {

//! Human readable titles of statistics groups defined by statistics_groups enum.
/** The order of this vector elements must correspond to the statistics_counters
    structure layout. **/
const char* StatGroupTitles[] = {
    "task objects", "tasks executed", "stealing attempts", "task proxies", "arena", "market", "priority ops", "prio ops details"
};

//! Human readable titles of statistics elements defined by statistics_counters struct.
/** The order of this vector elements must correspond to the statistics_counters
    structure layout (with NULLs interspersed to separate groups). **/
const char* StatFieldTitles[] = {
    /*task objects*/        "active", "freed", "big", NULL,
    /*tasks executed*/      "total", "w/o spawn", NULL,
    /*stealing attempts*/   "succeeded", "failed", "conflicts", "backoffs", NULL,
    /*task proxies*/        "mailed", "revoked", "stolen", "bypassed", "ignored", NULL,
    /*arena*/               "switches", "roundtrips", "avg.conc", "avg.allot", NULL,
    /*market*/              "roundtrips", NULL,
    /*priority ops*/        "ar.switch", "mkt.switch", "ar.reset", "ref.fixup", "avg.ar.pr", "avg.mkt.pr", NULL,
    /*prio ops details*/    "winnows", "reloads", "orphaned", "winnowed", "reloaded", NULL
};

//! Class for logging statistics
/** There should be only one instance of this class.
    Results are written to a file "statistics.txt" in tab-separated format. */
class statistics_logger {
public:
    statistics_logger () {
        __TBB_ASSERT( sg_end - 1 == 1 << (sizeof(StatGroupTitles)/sizeof(*StatGroupTitles) - 1), NULL );

        my_file = fopen("statistics.txt","w");
        if( !my_file )
            perror("fopen(\"statistics.txt\"\")");
        // Initialize groups dump layout info
        group_start_field[0] = 0;
        for ( size_t i = 0, j = 0; i < NumGroups; ++i, ++j ) {
            __TBB_ASSERT( StatFieldTitles[j], "Empty group occurred" );
            while ( StatFieldTitles[j] )
                ++j;
            group_start_field[i + 1] = j - i; // -i accounts for preceding NULL separators
        }
        __TBB_ASSERT( group_start_field[NumGroups] == statistics_counters::size(),
                      "Wrong number of elements in StatFieldTitles" );
        dump( "\n%-*s", IDColumnWidth, "");
        process_groups( &statistics_logger::print_group_title );
        dump( "%-*s", IDColumnWidth, "ID");
        process_groups( &statistics_logger::print_field_titles );
    }

    ~statistics_logger () { fclose(my_file); }

    void record( const statistics_counters& c, size_t id ) {
        spin_mutex::scoped_lock lock(my_mutex);
        counters_to_dump = &c;
#if __TBB_STATISTICS_TOTALS_ONLY
        if ( id == arena_counters_total ) {
            dump( "%-*s", IDColumnWidth, "Tot" );
            process_groups( &statistics_logger::print_field_values );
        }
#else /* !__TBB_STATISTICS_TOTALS_ONLY */
        const char* idString = NULL;
        switch ( id ) {
        case 0:
            idString = "M"; break;
        case workers_counters_total:
            idString = "Wtot"; break;
        case arena_counters_total:
            idString = "Tot"; break;
        default:
            dump( "W%-*u", IDColumnWidth - 1, id );
        }
        if ( idString )
            dump( "%-*s", IDColumnWidth, idString );
        process_groups( &statistics_logger::print_field_values );
#endif /* !__TBB_STATISTICS_TOTALS_ONLY */
    }
private:
    static const size_t IDColumnWidth = 5;
    static const size_t StatisticsColumnWidth = 10;
    static const size_t NumGroups = sizeof(StatGroupTitles)/sizeof(char*);

    //! File into which statistics are written.
    FILE* my_file;
    //! Mutex that serializes accesses to my_file
    spin_mutex my_mutex;
    //! Indices of the each group's first field in statistics_counters struct.
    /** An extra element is used to track the total number of statistics fields. **/
    size_t group_start_field[NumGroups + 1];
    //! Currently processed set of counters.
    const statistics_counters* counters_to_dump;

    static const size_t NumFields = sizeof(StatFieldTitles)/sizeof(*StatFieldTitles) - NumGroups;
    bool averages_fields[NumFields];

    void dump ( char const* fmt, ... ) {
        va_list args;
        if ( my_file ) {
            va_start( args, fmt );
            vfprintf( my_file, fmt, args );
            va_end( args );
        }
#if __TBB_STATISTICS_STDOUT
        va_start( args, fmt );
        vprintf( fmt, args );
        va_end( args );
#endif
    }

    void process_groups ( void (statistics_logger::*per_group_action)(size_t group_idx) ) {
        for ( size_t i = 0, group_flag = 1; i < NumGroups; ++i, group_flag <<= 1 ) {
            __TBB_ASSERT( group_flag < sg_end, "StatGroupTitles contents is incompatible with statistics_groups definition" );
            if ( __TBB_ActiveStatisticsGroups & group_flag )
                (this->*per_group_action)( i );
        }
        dump( "\n" );
    }

    void print_group_title ( size_t group_idx ) {
        dump( "%-*s", (group_start_field[group_idx + 1] - group_start_field[group_idx]) * (StatisticsColumnWidth + 1),
                        StatGroupTitles[group_idx] );
    }

    void print_field_titles ( size_t group_idx ) {
        // +group_idx accounts for preceding NULL separators
        size_t i = group_start_field[group_idx] + group_idx;
        while ( StatFieldTitles[i] ) {
            averages_fields[i - group_idx] = strncmp(StatFieldTitles[i], "avg.", 4) == 0;
            dump( "%-*s ", StatisticsColumnWidth, StatFieldTitles[i++] );
        }
    }

    void print_field_values ( size_t group_idx ) {
        size_t begin = group_start_field[group_idx],
               end = group_start_field[group_idx + 1];
        for ( size_t i = begin; i < end; ++i ) {
            if ( averages_fields[i] )
                dump( "%-*.2f ", StatisticsColumnWidth, (double)counters_to_dump->field(i)/counters_to_dump->tasks_executed );
            else
                dump( "%-*ld ", StatisticsColumnWidth, counters_to_dump->field(i) );
        }
    }
}; // class statistics_logger

static statistics_logger the_statistics;

void dump_statistics ( const statistics_counters& c, size_t id ) {
    the_statistics.record(c, id);
}

} // namespace internal
} // namespace tbb

#endif /* __TBB_STATISTICS */
