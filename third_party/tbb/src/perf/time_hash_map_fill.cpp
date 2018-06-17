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

// configuration:

// Size of final table (must be multiple of STEP_*)
int MAX_TABLE_SIZE = 2000000;

// Specify list of unique percents (5-30,100) to test against. Max 10 values
#define UNIQUE_PERCENTS PERCENT(5); PERCENT(10); PERCENT(20); PERCENT(30); PERCENT(100)

#define SECONDS_RATIO 1000000 // microseconds

// enable/disable tests for:
#define BOX1 "CHMap"
#define BOX1TEST ValuePerSecond<Uniques<tbb::concurrent_hash_map<int,int> >, SECONDS_RATIO>
#define BOX1HEADER "tbb/concurrent_hash_map.h"

// enable/disable tests for:
#define BOX2 "CUMap"
#define BOX2TEST ValuePerSecond<Uniques<tbb::concurrent_unordered_map<int,int> >, SECONDS_RATIO>
#define BOX2HEADER "tbb/concurrent_unordered_map.h"

// enable/disable tests for:
//#define BOX3 "OLD"
#define BOX3TEST ValuePerSecond<Uniques<tbb::concurrent_hash_map<int,int> >, SECONDS_RATIO>
#define BOX3HEADER "tbb/concurrent_hash_map-5468.h"

#define TBB_USE_THREADING_TOOLS 0
//////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <math.h>
#include "tbb/tbb_stddef.h"
#include <vector>
#include <map>
// needed by hash_maps
#include <stdexcept>
#include <iterator>
#include <algorithm>    // Need std::swap
#include <utility>      // Need std::pair
#include <cstring>      // Need std::memset
#include <typeinfo>
#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_allocator.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/aligned_space.h"
#include "tbb/atomic.h"
#define __TBB_concurrent_unordered_set_H
#include "tbb/internal/_concurrent_unordered_impl.h"
#undef __TBB_concurrent_unordered_set_H
// for test
#include "tbb/spin_mutex.h"
#include "time_framework.h"


using namespace tbb;
using namespace tbb::internal;

/////////////////////////////////////////////////////////////////////////////////////////
// Input data built for test
int *Data;

// Main test class used to run the timing tests. All overridden methods are called by the framework
template<typename TableType>
struct Uniques : TesterBase {
    TableType Table;
    int n_items;

    // Initializes base class with number of test modes
    Uniques() : TesterBase(2), Table(MaxThread*16) {
        //Table->max_load_factor(1); // add stub into hash_map to uncomment it
    }
    ~Uniques() {}

    // Returns name of test mode specified by number
    /*override*/ std::string get_name(int testn) {
        if(testn == 1) return "find";
        return "insert";
    }

    // Informs the class that value and threads number become known
    /*override*/ void init() {
        n_items = value/threads_count; // operations
    }

    // Informs the class that the test mode for specified thread is about to start
    /*override*/ void test_prefix(int testn, int t) {
        barrier->wait();
        if(Verbose && !t && testn) printf("%s: inserted %u, %g%% of operations\n", tester_name, unsigned(Table.size()), 100.0*Table.size()/(value*testn));
    }

    // Executes test mode for a given thread. Return value is ignored when used with timing wrappers.
    /*override*/ double test(int testn, int t)
    {
        if( testn == 0 ) { // do insertions
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                Table.insert( std::make_pair(Data[i],t) );
            }
        } else { // do last finds
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                size_t c =
                    Table.count( Data[i] );
                ASSERT( c == 1, NULL ); // must exist
            }
        }
        return 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////
#undef max
#include <limits>

// Using BOX declarations from configuration
#include "time_sandbox.h"

int rounds = 0;
// Prepares the input data for given unique percent
void execute_percent(test_sandbox &the_test, int p) {
    int input_size = MAX_TABLE_SIZE*100/p;
    Data = new int[input_size];
    int uniques = p==100?std::numeric_limits<int>::max() : MAX_TABLE_SIZE;
    ASSERT(p==100 || p <= 30, "Function is broken for %% > 30 except for 100%%");
    for(int i = 0; i < input_size; i++)
        Data[i] = (rand()*rand())%uniques;
    for(int t = MinThread; t <= MaxThread; t++)
        the_test.factory(input_size, t); // executes the tests specified in BOX-es for given 'value' and threads
    the_test.report.SetRoundTitle(rounds++, "%d%%", p);
}
#define PERCENT(x) execute_percent(the_test, x)

int main(int argc, char* argv[]) {
    if(argc>1) Verbose = true;
    //if(argc>2) ExtraVerbose = true;
    MinThread = 1; MaxThread = task_scheduler_init::default_num_threads();
    ParseCommandLine( argc, argv );
    if(getenv("TABLE_SIZE"))
        MAX_TABLE_SIZE = atoi(getenv("TABLE_SIZE"));

    ASSERT(tbb_allocator<int>::allocator_type() == tbb_allocator<int>::scalable, "expecting scalable allocator library to be loaded. Please build it by:\n\t\tmake tbbmalloc");
    // Declares test processor
    test_sandbox the_test("time_hash_map_fill"/*, StatisticsCollector::ByThreads*/);
    srand(10101);
    UNIQUE_PERCENTS; // test the percents
    the_test.report.SetTitle("Operations per microsecond");
    the_test.report.SetRunInfo("Items", MAX_TABLE_SIZE);
    the_test.report.Print(StatisticsCollector::HTMLFile|StatisticsCollector::ExcelXML); // Write files
    return 0;
}
