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

//#define DO_SCALABLEALLOC

#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include "tbb/tbb_stddef.h"
#include "tbb/spin_mutex.h"
#ifdef DO_SCALABLEALLOC
#include "tbb/scalable_allocator.h"
#endif
#include "tbb/concurrent_vector.h"
#include "tbb/tbb_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "tbb/blocked_range.h"
#define HARNESS_CUSTOM_MAIN 1
#include "../test/harness.h"
//#include "harness_barrier.h"
#include "../test/harness_allocator.h"
#define STATISTICS_INLINE
#include "statistics.h"

using namespace tbb;
bool ExtraVerbose = false;

class Timer {
    tbb::tick_count tick;
public:
    Timer() { tick = tbb::tick_count::now(); }
    double get_time()  { return (tbb::tick_count::now() - tick).seconds(); }
    double diff_time(const Timer &newer) { return (newer.tick - tick).seconds(); }
    double mark_time() { tick_count t1(tbb::tick_count::now()), t2(tick); tick = t1; return (t1 - t2).seconds(); }
    double mark_time(const Timer &newer) { tick_count t(tick); tick = newer.tick; return (tick - t).seconds(); }
};

/************************************************************************/
/* TEST1                                                                */
/************************************************************************/
#define mk_vector_test1(v, a) vector_test1<v<Timer, static_counting_allocator<a<Timer> > >, v<double, static_counting_allocator<a<double> > > >
template<class timers_vector_t, class values_vector_t>
class vector_test1 {
    const char *mode;
    StatisticsCollector &stat;
    StatisticsCollector::TestCase key[16];

public:
    vector_test1(const char *m, StatisticsCollector &s)  :  mode(m), stat(s) {}

    vector_test1 &operator()(size_t len) {
        if(Verbose) printf("test1<%s>(%u): collecting timing statistics\n", mode, unsigned(len));
        __TBB_ASSERT(sizeof(Timer) == sizeof(double), NULL);
        static const char *test_names[] = {
            "b)creation wholly",
            "a)creation by push",
            "c)operation time per item",
            0 };
        for(int i = 0; test_names[i]; ++i) key[i] = stat.SetTestCase(test_names[i], mode, len);

        Timer timer0; timers_vector_t::allocator_type::init_counters();
        timers_vector_t tv(len);
        Timer timer1; values_vector_t::allocator_type::init_counters();
        values_vector_t dv;
        for (size_t i = 0; i < len; ++i)
            dv.push_back( i );
        Timer timer2;
        for (size_t i = 0; i < len; ++i)
        {
            dv[len-i-1] = timer0.diff_time(tv[i]);
            tv[i].mark_time();
        }
        stat.AddStatisticValue( key[2], "1total, ms", "%.3f", timer2.get_time()*1e+3 );
        stat.AddStatisticValue( key[1], "1total, ms", "%.3f", timer1.diff_time(timer2)*1e+3 );
        stat.AddStatisticValue( key[0], "1total, ms", "%.3f", timer0.diff_time(timer1)*1e+3 );
        //allocator statistics
        stat.AddStatisticValue( key[0], "2total allocations", "%d", int(timers_vector_t::allocator_type::allocations) );
        stat.AddStatisticValue( key[1], "2total allocations", "%d", int(values_vector_t::allocator_type::allocations) );
        stat.AddStatisticValue( key[2], "2total allocations", "%d",  0);
        stat.AddStatisticValue( key[0], "3total alloc#items", "%d", int(timers_vector_t::allocator_type::items_allocated) );
        stat.AddStatisticValue( key[1], "3total alloc#items", "%d", int(values_vector_t::allocator_type::items_allocated) );
        stat.AddStatisticValue( key[2], "3total alloc#items", "%d",  0);
        //remarks
        stat.AddStatisticValue( key[0], "9note", "segment creation time, us:");
        stat.AddStatisticValue( key[2], "9note", "average op-time per item, us:");
        Timer last_timer(timer2); double last_value = 0;
        for (size_t j = 0, i = 2; i < len; i *= 2, j++) {
            stat.AddRoundResult( key[0], (dv[len-i-1]-last_value)*1e+6 );
            last_value = dv[len-i-1];
            stat.AddRoundResult( key[2], last_timer.diff_time(tv[i])/double(i)*1e+6 );
            last_timer = tv[i];
            stat.SetRoundTitle(j, i);
        }
        tv.clear(); dv.clear();
        //__TBB_ASSERT(timers_vector_t::allocator_type::items_allocated == timers_vector_t::allocator_type::items_freed, NULL);
        //__TBB_ASSERT(values_vector_t::allocator_type::items_allocated == values_vector_t::allocator_type::items_freed, NULL);
    	return *this;
    }
};

/************************************************************************/
/* TEST2                                                                */
/************************************************************************/
#define mk_vector_test2(v, a) vector_test2<v<size_t, a<size_t> > >
template<class vector_t>
class vector_test2 {
    const char *mode;
    static const int ntrial = 10;
    StatisticsCollector &stat;

public:
    vector_test2(const char *m, StatisticsCollector &s)  :  mode(m), stat(s) {}

    vector_test2 &operator()(size_t len) {
        if(Verbose) printf("test2<%s>(%u): performing standard transformation sequence on vector\n", mode, unsigned(len));
        StatisticsCollector::TestCase init_key = stat.SetTestCase("allocate", mode, len);
        StatisticsCollector::TestCase fill_key = stat.SetTestCase("fill", mode, len);
        StatisticsCollector::TestCase proc_key = stat.SetTestCase("process", mode, len);
        StatisticsCollector::TestCase full_key = stat.SetTestCase("total time", mode, len);
        for (int i = 0; i < ntrial; i++) {
            Timer timer0;
            vector_t v1(len);
            vector_t v2(len);
            Timer timer1;
            std::generate(v1.begin(), v1.end(), values(0));
            std::generate(v2.begin(), v2.end(), values(size_t(-len)));
            Timer timer2;
            std::reverse(v1.rbegin(), v1.rend());
            std::inner_product(v1.begin(), v1.end(), v2.rbegin(), 1);
            std::sort(v1.rbegin(), v1.rend());
            std::sort(v2.rbegin(), v2.rend());
            std::set_intersection(v1.begin(), v1.end(), v2.rbegin(), v2.rend(), v1.begin());
            Timer timer3;
            stat.AddRoundResult( proc_key, timer2.diff_time(timer3)*1e+3 );
            stat.AddRoundResult( fill_key, timer1.diff_time(timer2)*1e+3 );
            stat.AddRoundResult( init_key, timer0.diff_time(timer1)*1e+3 );
            stat.AddRoundResult( full_key, timer0.diff_time(timer3)*1e+3 );
        }
        stat.SetStatisticFormula("1Average", "=AVERAGE(ROUNDS)");
        stat.SetStatisticFormula("2+/-", "=(MAX(ROUNDS)-MIN(ROUNDS))/2");
        return *this;
    }

    class values
    {
        size_t value;
    public:
        values(size_t i) : value(i) {}
        size_t operator()() {
            return value++%(1|(value^55));
        }
    };
};

/************************************************************************/
/* TEST3                                                                */
/************************************************************************/
#define mk_vector_test3(v, a) vector_test3<v<char, local_counting_allocator<a<char>, size_t > > >
template<class vector_t>
class vector_test3 {
    const char *mode;
    StatisticsCollector &stat;

public:
    vector_test3(const char *m, StatisticsCollector &s)  :  mode(m), stat(s) {}

    vector_test3 &operator()(size_t len) {
        if(Verbose) printf("test3<%s>(%u): collecting allocator statistics\n", mode, unsigned(len));
        static const size_t sz = 1024;
        vector_t V[sz];
        StatisticsCollector::TestCase vinst_key = stat.SetTestCase("instances number", mode, len);
        StatisticsCollector::TestCase count_key = stat.SetTestCase("allocations count", mode, len);
        StatisticsCollector::TestCase items_key = stat.SetTestCase("allocated items", mode, len);
        //stat.ReserveRounds(sz-1);
        for (size_t c = 0, i = 0, s = sz/2; s >= 1 && i < sz; s /= 2, c++)
        {
            const size_t count = c? 1<<(c-1) : 0;
            for (size_t e = i+s; i < e; i++) {
                //if(count >= 16) V[i].reserve(count);
                for (size_t j = 0; j < count; j++)
                    V[i].push_back(j);
            }
            stat.SetRoundTitle ( c, count );
            stat.AddRoundResult( vinst_key, s );
            stat.AddRoundResult( count_key, V[i-1].get_allocator().allocations );
            stat.AddRoundResult( items_key, V[i-1].get_allocator().items_allocated );
        }
        return *this;
    }
};

/************************************************************************/
/* TYPES SET FOR TESTS                                                  */
/************************************************************************/
#define types_set(n, title, op) { StatisticsCollector Collector("time_vector"#n); Collector.SetTitle title; \
    {mk_vector_test##n(tbb::concurrent_vector, tbb::cache_aligned_allocator) ("TBB:NFS", Collector)op;} \
    {mk_vector_test##n(tbb::concurrent_vector, tbb::tbb_allocator)           ("TBB:TBB", Collector)op;} \
    {mk_vector_test##n(tbb::concurrent_vector, std::allocator)               ("TBB:STD", Collector)op;} \
    {mk_vector_test##n(std::vector, tbb::cache_aligned_allocator)            ("STL:NFS", Collector)op;} \
    {mk_vector_test##n(std::vector, tbb::tbb_allocator)                      ("STL:TBB", Collector)op;} \
    {mk_vector_test##n(std::vector, std::allocator)                          ("STL:STD", Collector)op;} \
    Collector.Print(StatisticsCollector::Stdout|StatisticsCollector::HTMLFile|StatisticsCollector::ExcelXML); }


/************************************************************************/
/* MAIN DRIVER                                                          */
/************************************************************************/
int main(int argc, char* argv[]) {
	if(argc>1) Verbose = true;
	if(argc>2) ExtraVerbose = true;
    MinThread = 0; MaxThread = 500000; // use in another meaning - test#:problem size
    ParseCommandLine( argc, argv );

    ASSERT(tbb_allocator<int>::allocator_type() == tbb_allocator<int>::scalable, "expecting scalable allocator library to be loaded");
    
    if(!MinThread || MinThread == 1)
        types_set(1, ("Vectors performance test #1 for %d", MaxThread), (MaxThread) )
    if(!MinThread || MinThread == 2)
        types_set(2, ("Vectors performance test #2 for %d", MaxThread), (MaxThread) )
    if(!MinThread || MinThread == 3)
        types_set(3, ("Vectors performance test #3 for %d", MaxThread), (MaxThread) )

    if(!Verbose) printf("done\n");
    return 0;
}

