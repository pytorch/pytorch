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

#ifndef __TIME_FRAMEWORK_H__
#define __TIME_FRAMEWORK_H__

#include <cstdlib>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include "tbb/tbb_stddef.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#define HARNESS_CUSTOM_MAIN 1
#include "../test/harness.h"
#include "../test/harness_barrier.h"
#define STATISTICS_INLINE
#include "statistics.h"

#ifndef ARG_TYPE
typedef intptr_t arg_t;
#else
typedef ARG_TYPE arg_t;
#endif

class Timer {
    tbb::tick_count tick;
public:
    Timer() { tick = tbb::tick_count::now(); }
    double get_time()  { return (tbb::tick_count::now() - tick).seconds(); }
    double diff_time(const Timer &newer) { return (newer.tick - tick).seconds(); }
    double mark_time() { tbb::tick_count t1(tbb::tick_count::now()), t2(tick); tick = t1; return (t1 - t2).seconds(); }
    double mark_time(const Timer &newer) { tbb::tick_count t(tick); tick = newer.tick; return (tick - t).seconds(); }
};

class TesterBase /*: public tbb::internal::no_copy*/ {
protected:
    friend class TestProcessor;
    friend class TestRunner;

    //! it is barrier for synchronizing between threads
    Harness::SpinBarrier *barrier;
    
    //! number of tests per this tester
    const int tests_count;
    
    //! number of threads to operate
    int threads_count;

    //! some value for tester
    arg_t value;

    //! tester name
    const char *tester_name;

    // avoid false sharing
    char pad[128 - sizeof(arg_t) - sizeof(int)*2 - sizeof(void*)*2 ];

public:
    //! init tester base. @arg ntests is number of embedded tests in this tester.
    TesterBase(int ntests)
        : barrier(NULL), tests_count(ntests)
    {}
    virtual ~TesterBase() {}

    //! internal function
    void base_init(arg_t v, int t, Harness::SpinBarrier &b) {
        threads_count = t;
        barrier = &b;
        value = v;
        init();
    }

    //! optionally override to init after value and threads count were set.
    virtual void init() { }

    //! Override to provide your names
    virtual std::string get_name(int testn) {
        return Format("test %d", testn);
    }

    //! optionally override to init test mode just before execution for a given thread number.
    virtual void test_prefix(int testn, int threadn) { }

    //! Override to provide main test's entry function returns a value to record
    virtual value_t test(int testn, int threadn) = 0;

    //! Type of aggregation from results of threads
    enum result_t {
        SUM, AVG, MIN, MAX
    };

    //! Override to change result type for the test. Return postfix for test name or 0 if result type is not needed.
    virtual const char *get_result_type(int /*testn*/, result_t type) const {
        return type == AVG ? "" : 0; // only average result by default
    }
};

/*****
a user's tester concept:

class tester: public TesterBase {
public:
    //! init tester with known amount of work
    tester() : TesterBase(<user-specified tests count>) { ... }

    //! run a test with sequental number @arg test_number for @arg thread.
    / *override* / value_t test(int test_number, int thread);
};

******/

template<typename Tester, int scale = 1>
class TimeTest : public Tester {
    /*override*/ value_t test(int testn, int threadn) {
        Timer timer;
        Tester::test(testn, threadn);
        return timer.get_time() * double(scale);
    }
};

template<typename Tester>
class NanosecPerValue : public Tester {
    /*override*/ value_t test(int testn, int threadn) {
        Timer timer;
        Tester::test(testn, threadn);
        // return time (ns) per value
        return timer.get_time()*1e+9/double(Tester::value);
    }
};

template<typename Tester, int scale = 1>
class ValuePerSecond : public Tester {
    /*override*/ value_t test(int testn, int threadn) {
        Timer timer;
        Tester::test(testn, threadn);
        // return value per seconds/scale
        return double(Tester::value)/(timer.get_time()*scale);
    }
};

template<typename Tester, int scale = 1>
class NumberPerSecond : public Tester {
    /*override*/ value_t test(int testn, int threadn) {
        Timer timer;
        Tester::test(testn, threadn);
        // return a scale per seconds
        return double(scale)/timer.get_time();
    }
};

// operate with single tester
class TestRunner {
    friend class TestProcessor;
    friend struct RunArgsBody;
    TestRunner(const TestRunner &); // don't copy

    const char *tester_name;
    StatisticsCollector *stat;
    std::vector<std::vector<StatisticsCollector::TestCase> > keys;

public:
    TesterBase &tester;

    template<typename Test>
    TestRunner(const char *name, Test *test)
        : tester_name(name), tester(*static_cast<TesterBase*>(test))
    {
        test->tester_name = name;
    }
    
    ~TestRunner() { delete &tester; }

    void init(arg_t value, int threads, Harness::SpinBarrier &barrier, StatisticsCollector *s) {
        tester.base_init(value, threads, barrier);
        stat = s;
        keys.resize(tester.tests_count);
        for(int testn = 0; testn < tester.tests_count; testn++) {
            keys[testn].resize(threads);
            std::string test_name(tester.get_name(testn));
            for(int threadn = 0; threadn < threads; threadn++)
                keys[testn][threadn] = stat->SetTestCase(tester_name, test_name.c_str(), threadn);
        }
    }

    void run_test(int threadn) {
        for(int testn = 0; testn < tester.tests_count; testn++) {
            tester.test_prefix(testn, threadn);
            tester.barrier->wait();                                 // <<<<<<<<<<<<<<<<< Barrier before running test mode
            value_t result = tester.test(testn, threadn);
            stat->AddRoundResult(keys[testn][threadn], result);
        }
    }

    void post_process(StatisticsCollector &report) {
        const int threads = tester.threads_count;
        for(int testn = 0; testn < tester.tests_count; testn++) {
            size_t coln = keys[testn][0].getResults().size()-1;
            value_t rsum = keys[testn][0].getResults()[coln];
            value_t rmin = rsum, rmax = rsum;
            for(int threadn = 1; threadn < threads; threadn++) {
                value_t result = keys[testn][threadn].getResults()[coln];
                rsum += result; // for both SUM or AVG
                if(rmin > result) rmin = result;
                if(rmax < result) rmax = result;
            }
            std::string test_name(tester.get_name(testn));
            const char *rname = tester.get_result_type(testn, TesterBase::SUM);
            if( rname ) {
                report.SetTestCase(tester_name, (test_name+rname).c_str(), threads);
                report.AddRoundResult(rsum);
            }
            rname = tester.get_result_type(testn, TesterBase::MIN);
            if( rname ) {
                report.SetTestCase(tester_name, (test_name+rname).c_str(), threads);
                report.AddRoundResult(rmin);
            }
            rname = tester.get_result_type(testn, TesterBase::AVG);
            if( rname ) {
                report.SetTestCase(tester_name, (test_name+rname).c_str(), threads);
                report.AddRoundResult(rsum / threads);
            }
            rname = tester.get_result_type(testn, TesterBase::MAX);
            if( rname ) {
                report.SetTestCase(tester_name, (test_name+rname).c_str(), threads);
                report.AddRoundResult(rmax);
            }
        }
    }
};

struct RunArgsBody {
    const vector<TestRunner*> &run_list;
    RunArgsBody(const vector<TestRunner*> &a) : run_list(a) { }
#ifndef __TBB_parallel_for_H
    void operator()(int thread) const {
#else
    void operator()(const tbb::blocked_range<int> &r) const {
        ASSERT( r.begin() + 1 == r.end(), 0);
        int thread = r.begin();
#endif
        for(size_t i = 0; i < run_list.size(); i++)
            run_list[i]->run_test(thread);
    }
};

//! Main test processor.
/** Override or use like this:
 class MyTestCollection : public TestProcessor {
    void factory(arg_t value, int threads) {
        process( value, threads,
            run("my1", new tester<my1>() ),
            run("my2", new tester<my2>() ),
        end );
        if(value == threads)
            stat->Print();
    }
};
*/

class TestProcessor {
    friend class TesterBase;

    // <threads, collector>
    typedef std::map<int, StatisticsCollector *> statistics_collection;
    statistics_collection stat_by_threads;

protected:
    // Members
    const char *collection_name;
    // current stat
    StatisticsCollector *stat;
    // token
    size_t end;

public:
    StatisticsCollector report;

    // token of tests list
    template<typename Test>
    TestRunner *run(const char *name, Test *test) {
        return new TestRunner(name, test);
    }

    // iteration processing
    void process(arg_t value, int threads, ...) {
        // prepare items
        stat = stat_by_threads[threads];
        if(!stat) {
            stat_by_threads[threads] = stat = new StatisticsCollector((collection_name + Format("@%d", threads)).c_str(), StatisticsCollector::ByAlg);
            stat->SetTitle("Detailed log of %s running with %d threads.", collection_name, threads);
        }
        Harness::SpinBarrier barrier(threads);
        // init args
        va_list args; va_start(args, threads);
        vector<TestRunner*> run_list; run_list.reserve(16);
        while(true) {
            TestRunner *item = va_arg(args, TestRunner*);
            if( !item ) break;
            item->init(value, threads, barrier, stat);
            run_list.push_back(item);
        }
        va_end(args);
        std::ostringstream buf;
        buf << value;
        const size_t round_number = stat->GetRoundsCount();
        stat->SetRoundTitle(round_number, buf.str().c_str());
        report.SetRoundTitle(round_number, buf.str().c_str());
        // run them
#ifndef __TBB_parallel_for_H
        NativeParallelFor(threads, RunArgsBody(run_list));
#else
        tbb::parallel_for(tbb::blocked_range<int>(0,threads,1), RunArgsBody(run_list));
#endif
        // destroy args
        for(size_t i = 0; i < run_list.size(); i++) {
            run_list[i]->post_process(report);
            delete run_list[i];
        }
    }

public:
    TestProcessor(const char *name, StatisticsCollector::Sorting sort_by = StatisticsCollector::ByAlg)
        : collection_name(name), stat(NULL), end(0), report(collection_name, sort_by)
    { }

    ~TestProcessor() {
        for(statistics_collection::iterator i = stat_by_threads.begin(); i != stat_by_threads.end(); i++)
            delete i->second;
    }
};

#endif// __TIME_FRAMEWORK_H__
