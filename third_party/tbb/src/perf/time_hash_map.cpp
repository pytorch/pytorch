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
#define TBB_USE_THREADING_TOOLS 0

//! enable/disable std::map tests
#define STDTABLE 0

//! enable/disable old implementation tests (correct include file also)
#define OLDTABLE 0
#define OLDTABLEHEADER "tbb/concurrent_hash_map-5468.h"//-4329

//! enable/disable experimental implementation tests (correct include file also)
#define TESTTABLE 0
#define TESTTABLEHEADER "tbb/concurrent_unordered_map.h"

//! avoid erase()
#define TEST_ERASE 1

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

struct IntHashCompare {
    size_t operator() ( int x ) const { return x; }
    bool operator() ( int x, int y ) const { return x==y; }
    static long hash( int x ) { return x; }
    bool equal( int x, int y ) const { return x==y; }
};

namespace version_current {
    namespace tbb { using namespace ::tbb; namespace internal { using namespace ::tbb::internal; } }
    namespace tbb { namespace interface5 { using namespace ::tbb::interface5; namespace internal { using namespace ::tbb::interface5::internal; } } }
    #include "tbb/concurrent_hash_map.h"
}
typedef version_current::tbb::concurrent_hash_map<int,int> IntTable;

#if OLDTABLE
#undef __TBB_concurrent_hash_map_H
namespace version_base {
    namespace tbb { using namespace ::tbb; namespace internal { using namespace ::tbb::internal; } }
    namespace tbb { namespace interface5 { using namespace ::tbb::interface5; namespace internal { using namespace ::tbb::interface5::internal; } } }
    #include OLDTABLEHEADER
}
typedef version_base::tbb::concurrent_hash_map<int,int> OldTable;
#endif

#if TESTTABLE
#undef __TBB_concurrent_hash_map_H
namespace version_new {
    namespace tbb { using namespace ::tbb; namespace internal { using namespace ::tbb::internal; } }
    namespace tbb { namespace interface5 { using namespace ::tbb::interface5; namespace internal { using namespace ::tbb::interface5::internal; } } }
    #include TESTTABLEHEADER
}
typedef version_new::tbb::concurrent_unordered_map<int,int> TestTable;
#define TESTTABLE 1
#endif

///////////////////////////////////////

static const char *map_testnames[] = {
    "1.insert", "2.count1st", "3.count2nd", "4.insert-exists", "5.erase "
};

template<typename TableType>
struct TestTBBMap : TesterBase {
    TableType Table;
    int n_items;

    TestTBBMap() : TesterBase(4+TEST_ERASE), Table(MaxThread*4) {}
    void init() { n_items = value/threads_count; }

    std::string get_name(int testn) {
        return std::string(map_testnames[testn]);
    }

    double test(int test, int t)
    {
        switch(test) {
          case 0: // fill
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                Table.insert( std::make_pair(i,i) );
            }
            break;
          case 1: // work1
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                size_t c = Table.count( i );
                ASSERT( c == 1, NULL);
            }
            break;
          case 2: // work2
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                Table.count( i );
            }
            break;
          case 3: // work3
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                Table.insert( std::make_pair(i,i) );
            }
            break;
#if TEST_ERASE
          case 4: // clean
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                ASSERT( Table.erase( i ), NULL);
            }
#endif
        }
        return 0;
    }
};

template<typename M>
struct TestSTLMap : TesterBase {
    std::map<int, int> Table;
    M mutex;

    int n_items;
    TestSTLMap() : TesterBase(4+TEST_ERASE) {}
    void init() { n_items = value/threads_count; }

    std::string get_name(int testn) {
        return std::string(map_testnames[testn]);
    }

    double test(int test, int t)
    {
        switch(test) {
          case 0: // fill
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                typename M::scoped_lock with(mutex);
                Table[i] = 0;
            }
            break;
          case 1: // work1
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                typename M::scoped_lock with(mutex);
                size_t c = Table.count(i);
                ASSERT( c == 1, NULL);
            }
            break;
          case 2: // work2
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                typename M::scoped_lock with(mutex);
                Table.count(i);
            }
            break;
          case 3: // work3
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                typename M::scoped_lock with(mutex);
                Table.insert(std::make_pair(i,i));
            }
            break;
          case 4: // clean
            for(int i = t*n_items, e = (t+1)*n_items; i < e; i++) {
                typename M::scoped_lock with(mutex);
                Table.erase(i);
            }
        }
        return 0;
    }
};

class fake_mutex {
public:
    class scoped_lock {
        fake_mutex *p;

    public:
        scoped_lock() {}
        scoped_lock( fake_mutex &m ) { p = &m; }
        ~scoped_lock() { }
        void acquire( fake_mutex &m ) { p = &m; }
        void release() { }
    };
};

class test_hash_map : public TestProcessor {
public:
    test_hash_map() : TestProcessor("time_hash_map") {}
    void factory(int value, int threads) {
        if(Verbose) printf("Processing with %d threads: %d...\n", threads, value);
        process( value, threads,
#if STDTABLE
            run("std::map ", new NanosecPerValue<TestSTLMap<spin_mutex> >() ),
#endif
#if OLDTABLE
            run("old::hmap", new NanosecPerValue<TestTBBMap<OldTable> >() ),
#endif
            run("tbb::hmap", new NanosecPerValue<TestTBBMap<IntTable> >() ),
#if TESTTABLE
            run("new::hmap", new NanosecPerValue<TestTBBMap<TestTable> >() ),
#endif
        end );
        //stat->Print(StatisticsCollector::Stdout);
        //if(value >= 2097152) stat->Print(StatisticsCollector::HTMLFile);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    if(argc>1) Verbose = true;
    //if(argc>2) ExtraVerbose = true;
    MinThread = 1; MaxThread = task_scheduler_init::default_num_threads();
    ParseCommandLine( argc, argv );

    ASSERT(tbb_allocator<int>::allocator_type() == tbb_allocator<int>::scalable, "expecting scalable allocator library to be loaded. Please build it by:\n\t\tmake tbbmalloc");

    {
        test_hash_map the_test;
        for( int t=MinThread; t <= MaxThread; t++)
            for( int o=/*2048*/(1<<8)*8; o<2200000; o*=2 )
                the_test.factory(o, t);
        the_test.report.SetTitle("Nanoseconds per operation of (Mode) for N items in container (Name)");
        the_test.report.SetStatisticFormula("1AVG per size", "=AVERAGE(ROUNDS)");
        the_test.report.Print(StatisticsCollector::HTMLFile|StatisticsCollector::ExcelXML);
    }
    return 0;
}
