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

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"
#include "harness.h"
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include "tbb/task_scheduler_observer.h"

const size_t MB = 1024*1024;
const double BARRIER_TIMEOUT = 10.;

void TestStackSizeSimpleControl()
{
    {
        tbb::global_control s0(tbb::global_control::thread_stack_size, 1*MB);

        {
            tbb::global_control s1(tbb::global_control::thread_stack_size, 8*MB);

            ASSERT(8*MB == tbb::global_control::active_value(tbb::global_control::thread_stack_size), NULL);
        }
        ASSERT(1*MB == tbb::global_control::active_value(tbb::global_control::thread_stack_size), NULL);
    }
}

#include "harness_concurrency_tracker.h"
#include "tbb/task_scheduler_init.h"
#include <limits.h>   // for UINT_MAX

struct StackSizeRun: NoAssign {
    int                   num_threads;
    Harness::SpinBarrier *barr1, *barr2;

    StackSizeRun(int threads, Harness::SpinBarrier *b1, Harness::SpinBarrier *b2) :
        num_threads(threads), barr1(b1), barr2(b2) {}
    void operator()( int id ) const {
        tbb::global_control s1(tbb::global_control::thread_stack_size, (1+id)*MB);

        barr1->timed_wait(BARRIER_TIMEOUT);

        ASSERT(num_threads*MB == tbb::global_control::active_value(tbb::global_control::thread_stack_size), NULL);
        barr2->timed_wait(BARRIER_TIMEOUT);
    }
};

void TestStackSizeThreadsControl()
{
    int threads = 4;
    Harness::SpinBarrier barr1(threads), barr2(threads);
    NativeParallelFor( threads, StackSizeRun(threads, &barr1, &barr2) );
}

void RunWorkersLimited(int tsi_max_threads, size_t parallelism, bool wait)
{
    tbb::global_control s(tbb::global_control::max_allowed_parallelism, parallelism);
    // try both configuration with already sleeping workers and with not yet sleeping
    if (wait)
        Harness::Sleep(100);
    const size_t expected_threads = tsi_max_threads>0?
        min( (unsigned)tsi_max_threads, parallelism )
        : ( tbb::tbb_thread::hardware_concurrency()==1? 1 : parallelism );
    Harness::ExactConcurrencyLevel::check(expected_threads);
}

class blocking_task_scheduler_init {
    tbb::task_scheduler_init init;
public:
    blocking_task_scheduler_init(int num_threads = tbb::task_scheduler_init::automatic) : init(num_threads) {}
    ~blocking_task_scheduler_init() {
        bool ok = init.blocking_terminate(std::nothrow);
        ASSERT(ok, "blocking_terminate has failed");
    }
};

void TSI_and_RunWorkers(int tsi_max_threads, size_t parallelism, size_t max_value)
{
    blocking_task_scheduler_init tsi(tsi_max_threads);
    size_t active = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    ASSERT(active == max(2U, max_value), "active_value must not be changed by task_scheduler_init");
    RunWorkersLimited(tsi_max_threads, parallelism, /*wait=*/false);
}

#include "tbb/tbb_thread.h"

void TestWorkers(size_t curr_par)
{
    const size_t max_parallelism =
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    ASSERT(max(2U, tbb::tbb_thread::hardware_concurrency()) == max_parallelism, NULL);
    {
        const unsigned h_c = tbb::tbb_thread::hardware_concurrency();
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, curr_par);
        size_t v = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
        ASSERT(!curr_par || max((size_t)2, curr_par) == v, NULL);
        if (h_c > 1)
            TSI_and_RunWorkers(tbb::task_scheduler_init::automatic, min(h_c, curr_par), curr_par);
        if (curr_par) // do not call task_scheduler_init t(0);
            TSI_and_RunWorkers((int)curr_par, curr_par, curr_par);
        if (curr_par > 2) { // check that min(tsi, parallelism) is active
            TSI_and_RunWorkers((int)curr_par-1, curr_par, curr_par);
            TSI_and_RunWorkers((int)curr_par, curr_par-1, curr_par);
        }
        // check constrains on control's value: it can't be increased
        tbb::global_control c1(tbb::global_control::max_allowed_parallelism, curr_par+1);
        v = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
        if (curr_par)
            ASSERT(max(2U, curr_par) == v, "It's impossible to increase maximal parallelism.");
        else
            ASSERT(2 == v, NULL);
    }
    ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
           == max_parallelism,
           "max parallelism has been restored successfully after decreasing/increasing");
}

void TestWorkersConstraints() {
    const size_t max_parallelism =
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    blocking_task_scheduler_init tsi;
    if (max_parallelism > 3) {
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, max_parallelism-1);
        ASSERT(max_parallelism-1 ==
               tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism),
               "Allowed parallelism must be decreasable.");
        tbb::global_control c1(tbb::global_control::max_allowed_parallelism, max_parallelism-2);
        ASSERT(max_parallelism-2 ==
               tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism),
               "Allowed parallelism must be decreasable.");
    }
    const size_t limit_par = min(max_parallelism, 4U);
    // check that constrains are really met
    for (int wait=0; wait<2; wait++) {
        for (size_t num=2; num<limit_par; num++)
            RunWorkersLimited(tbb::task_scheduler_init::automatic, num, wait==1);
        for (size_t num=limit_par; num>1; num--)
            RunWorkersLimited(tbb::task_scheduler_init::automatic, num, wait==1);
    }
}

struct DummyBody {
    void operator()(int) const {
        __TBB_Pause(1);
    }
};

void RunParallelWork() {
    const int LOOP_ITERS = 10*1000;
    tbb::parallel_for(0, LOOP_ITERS, DummyBody(), tbb::simple_partitioner());
}

struct SetUseRun: NoAssign {
    Harness::SpinBarrier *barr;

    SetUseRun(Harness::SpinBarrier *b) : barr(b) {}
    void operator()( int id ) const {
        if (id == 0) {
            for (int i=0; i<10; i++) {
                blocking_task_scheduler_init tsi;
                RunParallelWork();
                barr->timed_wait(BARRIER_TIMEOUT);
            }
        } else {
            for (int i=0; i<10; i++) {
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, 8);
                barr->timed_wait(BARRIER_TIMEOUT);
            }
        }
    }
};

void TestConcurrentSetUseConcurrency()
{
    Harness::SpinBarrier barr(2);
    NativeParallelFor( 2, SetUseRun(&barr) );
}

// check number of workers after autoinitialization
void TestAutoInit()
{
    const size_t max_parallelism =
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    const unsigned expected_threads = tbb::tbb_thread::hardware_concurrency()==1?
        1 : (unsigned)max_parallelism;
    Harness::ExactConcurrencyLevel::check(expected_threads);
    ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
           == max_parallelism, "max_allowed_parallelism must not be changed after auto init");
    if (max_parallelism > 2) {
        // after autoinit it's possible to decrease workers number
        tbb::global_control s(tbb::global_control::max_allowed_parallelism, max_parallelism-1);
        Harness::ExactConcurrencyLevel::check(max_parallelism-1);
    }
}

// need this to use TRY_BAD_EXPR_ENABLED when TBB_USE_ASSERT is not defined
#undef TBB_USE_ASSERT
#define TBB_USE_ASSERT 1

#include "harness_bad_expr.h"

void TestInvalidParallelism()
{
#if TRY_BAD_EXPR_ENABLED
    const size_t max_parallelism =
        tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    {
        tbb::set_assertion_handler( AssertionFailureHandler );
        TRY_BAD_EXPR( tbb::global_control c(tbb::global_control::max_allowed_parallelism, 0),
                       "max_allowed_parallelism cannot be 0." );
        tbb::set_assertion_handler( ReportError );
        ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
               == max_parallelism, NULL);
    }
    {
        const size_t P = 2;
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, P);
        ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
               == P, NULL);
        tbb::set_assertion_handler( AssertionFailureHandler );
        TRY_BAD_EXPR( tbb::global_control cZ(tbb::global_control::max_allowed_parallelism, 0),
                      "max_allowed_parallelism cannot be 0." );
        tbb::set_assertion_handler( ReportError );
        ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
               == P, NULL);
    }
    ASSERT(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
           == max_parallelism, NULL);
#endif /* TRY_BAD_EXPR_ENABLED */
}

void TestTooBigStack()
{
#if __TBB_x86_32
    const size_t stack_sizes[] = {512*MB, 2*1024*MB, UINT_MAX};
#else
    const size_t stack_sizes[] = {512*MB, 2*1024*MB, UINT_MAX, 10LU*1024*MB};
#endif

#if __TBB_WIN8UI_SUPPORT
    size_t default_ss = tbb::global_control::active_value(tbb::global_control::thread_stack_size);
#endif
    for (unsigned i = 0; i<Harness::array_length(stack_sizes); i++) {
        // as no stack size setting for Windows Store* apps, skip it
#if TRY_BAD_EXPR_ENABLED && __TBB_x86_64 && (_WIN32 || _WIN64) && !__TBB_WIN8UI_SUPPORT
        if (stack_sizes[i] != (unsigned)stack_sizes[i]) {
            size_t curr_ss = tbb::global_control::active_value(tbb::global_control::thread_stack_size);
            tbb::set_assertion_handler( AssertionFailureHandler );
            TRY_BAD_EXPR( tbb::global_control s1(tbb::global_control::thread_stack_size, stack_sizes[i]), "Stack size is limited to unsigned int range" );
            tbb::set_assertion_handler( ReportError );
            ASSERT(curr_ss == tbb::global_control::active_value(tbb::global_control::thread_stack_size), "Changing of stack size is not expected.");
            continue;
        }
#endif
        tbb::global_control s1(tbb::global_control::thread_stack_size, stack_sizes[i]);
        size_t actual_stack_sz = tbb::global_control::active_value(tbb::global_control::thread_stack_size);
#if __TBB_WIN8UI_SUPPORT
        ASSERT(actual_stack_sz == default_ss, "It's ignored for Windows Store* apps");
#else
        ASSERT(actual_stack_sz==stack_sizes[i], NULL);
#endif
    }
}

struct ParallelForRun: NoAssign {
    int                   num_threads;
    Harness::SpinBarrier *barr1, *barr2;

    ParallelForRun(Harness::SpinBarrier *b1, Harness::SpinBarrier *b2) :
        barr1(b1), barr2(b2) {}
    void operator()( int /*id*/ ) const {
        barr1->timed_wait(BARRIER_TIMEOUT);
        RunParallelWork();
        barr2->timed_wait(BARRIER_TIMEOUT);
    }
};

class FFTask: public tbb::task {
    tbb::atomic<int> *counter;
    tbb::task* execute() __TBB_override {
        (*counter)++;
       return NULL;
   }
public:
    FFTask(tbb::atomic<int> *counter_) : counter(counter_) {}
};

class WaiterTask: public tbb::task {
    tbb::atomic<bool> *flag;
    tbb::task* execute() __TBB_override {
        while(!*flag)
            __TBB_Yield();
       return NULL;
   }
public:
    WaiterTask(tbb::atomic<bool> *flag_) : flag(flag_) {}
};

class WorkAndEnqueueTask: public tbb::task {
    tbb::atomic<int> *counter;
    tbb::atomic<bool> *signalToLeave;
    tbb::task* execute() __TBB_override {
        RunParallelWork();
        *signalToLeave = true;
        for (int i=0; i<ENQUEUE_TASKS; i++) {
            FFTask* t = new( tbb::task::allocate_root() ) FFTask(counter);
            tbb::task::enqueue(*t);
        }

        return NULL;
   }
public:
    static const int ENQUEUE_TASKS = 10;
    WorkAndEnqueueTask(tbb::atomic<int> *counter_, tbb::atomic<bool> *signal_)
        : counter(counter_), signalToLeave(signal_) {}
};

#if __TBB_TASK_PRIORITY
tbb::priority_t getPriorityByInt(int i) {
    return i%3==0? tbb::priority_low : (i%3==1? tbb::priority_normal :
                                        tbb::priority_high);
}
#endif

class FFTasksRun: NoAssign {
    void enqTasks(int id) const {
        for (int i=0; i<ITERS; i++) {
            FFTask* t = new( tbb::task::allocate_root() ) FFTask(cnt);
#if __TBB_TASK_PRIORITY
            tbb::priority_t p = getPriorityByInt(i+id);
            tbb::task::enqueue(*t, p);
#else
            tbb::internal::suppress_unused_warning(id);
            tbb::task::enqueue(*t);
#endif
        }
    }
public:
    static const int ITERS = 20;
    Harness::SpinBarrier *barr;
    tbb::atomic<int> *cnt;

    FFTasksRun(Harness::SpinBarrier *b, tbb::atomic<int> *c) :
        barr(b), cnt(c) {}
    void operator()(int id) const {
        if (id)
            enqTasks(id);
        barr->wait();
        if (!id)
            enqTasks(id);
    }
};

void TestTaskEnqueue()
{
    {
        blocking_task_scheduler_init tsi(20);
        tbb::atomic<int> flag;
        tbb::atomic<bool> taskDoneFlag;
        flag = 0;
        taskDoneFlag = false;

        for (int i=0; i<10; i++) {
            WaiterTask* w = new( tbb::task::allocate_root() ) WaiterTask(&taskDoneFlag);
            tbb::task::enqueue(*w);
        }
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
        taskDoneFlag = true;

        FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
        tbb::task::enqueue(*t);
        while(!flag)
            __TBB_Yield();
    }
    {
        blocking_task_scheduler_init tsi(1);
        tbb::atomic<int> flag;
        tbb::atomic<bool> taskDoneFlag;
        flag = 0;
        taskDoneFlag = false;

        WaiterTask* w = new( tbb::task::allocate_root() ) WaiterTask(&taskDoneFlag);
        tbb::task::enqueue(*w);
        taskDoneFlag = true;

        tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);

        FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
        tbb::task::enqueue(*t);
        while(!flag)
            __TBB_Yield();
    }
    {
        blocking_task_scheduler_init tsi(2);
        tbb::atomic<int> flag;
        flag = 0;

        tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);

        FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
        tbb::task::enqueue(*t);
        while(!flag)
            __TBB_Yield();
    }
    {
        blocking_task_scheduler_init tsi(2);
        tbb::atomic<int> flag;
        flag = 0;

        FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
        tbb::task::enqueue(*t);
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);

        while(!flag)
            __TBB_Yield();
    }

    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);

    { // check that enqueue() guarantee mandatory parallelism
        blocking_task_scheduler_init tsi(1);
        tbb::atomic<int> flag;
        flag = 0;

        FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
        tbb::task::enqueue(*t);
        while(!flag)
            __TBB_Yield();
    }
    {
        tbb::atomic<int> flag;
        flag = 0;
        {
            blocking_task_scheduler_init tsi(1);

            for (int i=0; i<10; i++) {
                FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
#if __TBB_TASK_PRIORITY
                const tbb::priority_t p = getPriorityByInt(i);
                tbb::task::enqueue(*t, p);
#else
                tbb::task::enqueue(*t);
#endif
            }
        }
        ASSERT(flag==10, "The tasks must be terminated when task_scheduler_init destroyed.");
    }
    const unsigned threads = 2;
    {
        blocking_task_scheduler_init tsi(1);
        Harness::SpinBarrier barr1(threads), barr2(threads);
        RunWorkersLimited(1, 1, false);

        NativeParallelFor( threads, ParallelForRun(&barr1, &barr2) );
    }

    tbb::atomic<int> counter;
    counter = 0;
    {
        blocking_task_scheduler_init tsi(1);
        Harness::SpinBarrier barr(threads);
        RunWorkersLimited(1, 1, false);

        NativeParallelFor( threads, FFTasksRun(&barr, &counter) );
    }
    ASSERT(counter == threads*FFTasksRun::ITERS, "All tasks must be done when task_scheduler_init destroyed.");
    counter = 0;
    { // an enqueued task can enqueue other tasks and calls parallel_for
        tbb::atomic<bool> signalToLeave;
        blocking_task_scheduler_init tsi(1);

        signalToLeave = false;
        WorkAndEnqueueTask *t = new( tbb::task::allocate_root() )
            WorkAndEnqueueTask(&counter, &signalToLeave);
        tbb::task::enqueue(*t);
        RunParallelWork();

        while (!signalToLeave)
            __TBB_Yield();
    }
    ASSERT(counter == WorkAndEnqueueTask::ENQUEUE_TASKS, "All tasks must be done when task_scheduler_init destroyed.");
}

class CountWorkersTask: public tbb::task {
    tbb::atomic<bool> *flag;
    // count unique worker threads
    static tbb::combinable<size_t> uniqThreads;

    tbb::task* execute() __TBB_override {
        uniqThreads.local() = 1;
        Harness::Sleep(10);
        *flag = 1;
        return NULL;
   }
public:
    CountWorkersTask(tbb::atomic<bool> *flag_) : flag(flag_) {}
    static size_t observedThreads() {
        return uniqThreads.combine(std::plus<size_t>());
    }
};

tbb::combinable<size_t> CountWorkersTask::uniqThreads;

tbb::atomic<int> activeArenas;

class ArenaObserver: public tbb::task_scheduler_observer {
public:
    ArenaObserver() : tbb::task_scheduler_observer(/*local=*/true) {
    }
    void on_scheduler_entry( bool worker ) __TBB_override {
        if (worker) {
            ++activeArenas;
        }
    }
    void on_scheduler_exit( bool worker ) __TBB_override {
        if (worker) {
            --activeArenas;
        }
    }
};

ArenaObserver observers[2];

struct ArenasObserveRun: NoAssign {
    Harness::SpinBarrier *barr;

    ArenasObserveRun(Harness::SpinBarrier *b) : barr(b) {}
    void operator()( int id ) const {
        observers[id].observe(true);
        ArenaObserver o;
        tbb::atomic<bool> flag;
        flag = false;

        CountWorkersTask* t = new( tbb::task::allocate_root() )
            CountWorkersTask(&flag);
        barr->wait();
        tbb::task::enqueue(*t);
        while(!flag)
            __TBB_Yield();
    }
};

struct ArenaRun: NoAssign {
    tbb::atomic<int> *counter;

    ArenaRun(tbb::atomic<int> *counter_) : counter(counter_) {}
    void operator()() const {
        (*counter)++;
    }
};

struct ArenaUserRun: NoAssign {
    static const int ENQUEUE_TASKS = 10;
    tbb::task_arena *arena;
    Harness::SpinBarrier *barr;
    tbb::atomic<int> *counter;

    ArenaUserRun(tbb::task_arena *a, Harness::SpinBarrier *b, tbb::atomic<int> *c) :
        arena(a), barr(b), counter(c)  {}
    void operator()( int id ) const {

        for (int i=0; i<ENQUEUE_TASKS; i++)
            arena->enqueue(ArenaRun(counter));
        barr->wait();
        if (!id)
            arena->terminate();
    }
};

void TestConcurrentArenas()
{
    Harness::SpinBarrier barrier(2);
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
    {
        blocking_task_scheduler_init tsi(2);
        ArenaObserver observer;
        observer.observe(true);

        Harness::ExactConcurrencyLevel::check(1); // must have 0 worker threads

        NativeParallelFor( 2, ArenasObserveRun(&barrier) );
        ASSERT(1 == CountWorkersTask::observedThreads(),
               "Single worker is expecting to serve mandatory parallelism.");
        while(activeArenas) // wait till single worker termination
            __TBB_Yield();

        // check that without mandatory parallelism, still have 0 worker threads
        Harness::ExactConcurrencyLevel::check(1);
    }
    tbb::atomic<int> counter;
    counter = 0;
    {
        blocking_task_scheduler_init tsi(1);
        tbb::task_arena arena(2);

        NativeParallelFor( 2, ArenaUserRun(&arena, &barrier, &counter) );
    }
    ASSERT(counter == 2*ArenaUserRun::ENQUEUE_TASKS, "All tasks must be done.");
}

void TestParallelismRestored()
{
    const int TASKS = 5;
    tbb::atomic<int> counter;
    counter = 0;
    {
        const int P = 4;
        blocking_task_scheduler_init tsi(P);
        {
            tbb::global_control s(tbb::global_control::max_allowed_parallelism, 1);
            Harness::ExactConcurrencyLevel::check(1);
            // create enforced concurrency in the arena
            for (int i=0; i<TASKS; i++) {
                FFTask* t = new( tbb::task::allocate_root() ) FFTask(&counter);
                tbb::task::enqueue(*t);
            }
        }
        // global control is off, check that concurrency P is available
        Harness::ExactConcurrencyLevel::check(P);
    }
    ASSERT(counter==TASKS, "The tasks must be executed at this point.");
}

class NoUnwantedEnforcedRun {
    Harness::SpinBarrier *globalBarrier;
public:
    NoUnwantedEnforcedRun(Harness::SpinBarrier *b) : globalBarrier(b) {}
    void operator()( int id ) const {
        Harness::SpinBarrier barr(1);

        tbb::combinable<size_t> uniqThreads;
        Harness::ExactConcurrencyLevel::check(1);
        globalBarrier->wait();
        if (id) {
            for (int i=0; i<20; i++) {
                Harness::ExactConcurrencyLevel::check(1); // no workers expected in the thread
            }
        } else {
            // create enforced concurrency in a separate thread, thus provoke enforced worker without
            // work to do to join arena with parallel_for
            for (int i=0; i<10; i++) {
                tbb::atomic<int> flag;
                flag = 0;
                FFTask* t = new( tbb::task::allocate_root() ) FFTask(&flag);
                tbb::task::enqueue(*t);
                Harness::ExactConcurrencyLevel::checkLessOrEqual(2, &uniqThreads);
                size_t seen = uniqThreads.combine(std::plus<size_t>());
                ASSERT(seen==1 || seen==2, NULL);
                while(!flag)
                    __TBB_Yield();
            }
        }
    }
};

// test that enforced concurrency from one thread doesn't affect another
void TestNoUnwantedEnforced()
{
    Harness::SpinBarrier barrier(2);
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
    blocking_task_scheduler_init tsi(4);
    NativeParallelFor( 2, NoUnwantedEnforcedRun(&barrier) );
}

class TestMultipleControlsRun {
    Harness::SpinBarrier *barrier;
public:
    TestMultipleControlsRun(Harness::SpinBarrier *b) : barrier(b) {}
    void operator()( int id ) const {
        barrier->wait();
        if (id) {
            {
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
                Harness::ExactConcurrencyLevel::check(1);
                barrier->wait();
            }
            Harness::ExactConcurrencyLevel::check(1);
            barrier->wait();
            {
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, 2);
                Harness::ExactConcurrencyLevel::check(1);
                barrier->wait();
                Harness::ExactConcurrencyLevel::check(2);
                barrier->wait();
            }
        } else {
            {
                Harness::ExactConcurrencyLevel::check(1);
                tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
                barrier->wait();
                Harness::ExactConcurrencyLevel::check(1);
                barrier->wait();
                Harness::ExactConcurrencyLevel::check(1);
                barrier->wait();
            }
            Harness::ExactConcurrencyLevel::check(2);
            barrier->wait();
        }
    }
};

// test that global controls from different thread with overlapping lifetime
// still keep parallelism under control
void TestMultipleControls()
{
    blocking_task_scheduler_init tsi(2); // to prevent autoinitialization
    Harness::SpinBarrier barrier(2);
    NativeParallelFor( 2, TestMultipleControlsRun(&barrier) );
}

// enqueued tasks with priority below current must not be forgotten,
// when enqueue enforced priority is enabled
void TestForgottenEnqueuedTasks()
{
    tbb::task_scheduler_init tsi(2);
    tbb::atomic<int> counter;
    tbb::atomic<bool> waitFlag;

    waitFlag = false;
    counter = 0;
    tbb::task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r.set_ref_count(3);
    for (int i=0; i<2; i++) {
        tbb::task &t = *new( r.allocate_child() ) WaiterTask(&waitFlag);
        tbb::task::spawn(t);
    }
    // all workers are occupied by blocked WaiterTask()
    FFTask* t = new( tbb::task::allocate_root() ) FFTask(&counter);
    tbb::task::enqueue(*t, tbb::priority_low);
    {
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
        waitFlag = true; // WaiterTask() done, workers ready to use
        while (!counter) // wait till FFTask() executed
            __TBB_Yield();
    }
    r.wait_for_all();
    tbb::task::destroy(r);
}

int TestMain()
{
    TestTaskEnqueue();
    TestConcurrentArenas();
    TestMultipleControls();
    TestNoUnwantedEnforced();
    const unsigned h_c = tbb::tbb_thread::hardware_concurrency();
    bool excessHC;
    {
        tbb::task_scheduler_init t(h_c+1);
        excessHC = Harness::ExactConcurrencyLevel::isEqual(h_c+1);
    }
    if (h_c>2)
        TestWorkers(h_c-1);
    if (excessHC)  // requires hardware concurrency +1, otherwise hangs
        TestWorkers(h_c+1);
    if (excessHC || h_c >= 2)
        TestWorkers(2);
    if (excessHC || h_c >= 3)
        TestWorkers(3);
    TestWorkersConstraints();
    TestConcurrentSetUseConcurrency();
    TestInvalidParallelism();
    TestAutoInit(); // auto-initialization done at this point

    size_t default_ss = tbb::global_control::active_value(tbb::global_control::thread_stack_size);
    ASSERT(default_ss, NULL);
#if !__TBB_WIN8UI_SUPPORT
    // it's impossible to change stack size for Windows Store* apps, so skip the tests
    TestStackSizeSimpleControl();
    TestStackSizeThreadsControl();
#endif
    TestTooBigStack();
    ASSERT(default_ss == tbb::global_control::active_value(tbb::global_control::thread_stack_size), NULL);
    return Harness::Done;
}
