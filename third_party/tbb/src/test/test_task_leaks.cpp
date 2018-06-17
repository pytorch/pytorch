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

/*  The test uses "single produces multiple consumers" (SPMC )pattern to check
    if the memory of the tasks stolen by consumer threads is returned to the
    producer thread and is reused.

    The test consists of a series of iterations, which execute a task tree.
    the test fails is the memory consumption is not stabilized during some
    number of iterations.

    After the memory consumption stabilized the memory state is perturbed by
    switching producer thread, and the check is repeated.
*/

#define HARNESS_DEFAULT_MIN_THREADS -1

#define  __TBB_COUNT_TASK_NODES 1
#include "harness_inject_scheduler.h"

#include "tbb/atomic.h"
#include "harness_assert.h"
#include <cstdlib>


// Test configuration parameters

//! Maximal number of test iterations
const int MaxIterations = 600;
//! Number of iterations during which the memory consumption must stabilize
const int AsymptoticRange = 100;
//! Number of times the memory state is perturbed to repeat the check
const int NumProducerSwitches = 2;
//! Number of iterations after which the success of producer switch is checked
const int ProducerCheckTimeout = 10;
//! Number of initial iteration used to collect statistics to be used in later checks
const int InitialStatsIterations = 20;
//! Inner iterations of RunTaskGenerators()
const int TaskGeneratorsIterations = TBB_USE_DEBUG ? 30 : 100;

tbb::atomic<int> Count;
tbb::atomic<tbb::task*> Exchanger;
tbb::internal::scheduler* Producer;

#include "tbb/task_scheduler_init.h"

#include "harness.h"

using namespace tbb;
using namespace tbb::internal;

class ChangeProducer: public tbb::task {
public:
    tbb::task* execute() __TBB_override {
        if( is_stolen_task() ) {
            Producer = internal::governor::local_scheduler();
        }
        return NULL;
    }
};

class TaskGenerator: public tbb::task {
    const int my_child_count;
    int my_depth;
public:
    TaskGenerator(int child_count, int d) : my_child_count(child_count), my_depth(d) {
        ASSERT(my_child_count>1, "The TaskGenerator should produce at least two children");
    }
    tbb::task* execute() __TBB_override {
        if( my_depth>0 ) {
            int child_count = my_child_count;
            scheduler* my_sched = internal::governor::local_scheduler();
            tbb::task& c  = *new( allocate_continuation() ) tbb::empty_task;
            c.set_ref_count( child_count );
            recycle_as_child_of(c);
            --child_count;
            if( Producer==my_sched ) {
                // produce a task and put it into Exchanger
                tbb::task* t = new( c.allocate_child() ) tbb::empty_task;
                --child_count;
                t = Exchanger.fetch_and_store(t);
                if( t ) spawn(*t);
            } else {
                tbb::task* t = Exchanger.fetch_and_store(NULL);
                if( t ) spawn(*t);
            }
            while( child_count ) {
                tbb::task* t = new( c.allocate_child() ) TaskGenerator(my_child_count, my_depth-1);
                if( my_depth >4 ) enqueue(*t);
                else              spawn(*t);
                --child_count;
            }
            --my_depth;
            return this;
        } else {
            tbb::task* t = Exchanger.fetch_and_store(NULL);
            if( t ) spawn(*t);
            return NULL;
        }
    }
};

#include "harness_memory.h"
#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
    // VS2008/VC9 seems to have an issue
    #pragma warning( push )
    #pragma warning( disable: 4985 )
#endif
#include <math.h>
#if _MSC_VER==1500 && !defined(__INTEL_COMPILER)
    #pragma warning( pop )
#endif

void RunTaskGenerators( bool switchProducer = false, bool checkProducer = false ) {
    if( switchProducer )
        Producer = NULL;
    tbb::task* dummy_root = new( tbb::task::allocate_root() ) tbb::empty_task;
    dummy_root->set_ref_count( 2 );
    // If no producer, start elections; some worker will take the role
    if( Producer )
        tbb::task::spawn( *new( dummy_root->allocate_child() ) tbb::empty_task );
    else
        tbb::task::spawn( *new( dummy_root->allocate_child() ) ChangeProducer );
    if( checkProducer && !Producer )
        REPORT("Warning: producer has not changed after 10 attempts; running on a single core?\n");
    for( int j=0; j<TaskGeneratorsIterations; ++j ) {
        if( j&1 ) {
            tbb::task& t = *new( tbb::task::allocate_root() ) TaskGenerator(/*child_count=*/4, /*depth=*/6);
            tbb::task::spawn_root_and_wait(t);
        } else {
            tbb::task& t = *new (tbb::task::allocate_additional_child_of(*dummy_root))
                                TaskGenerator(/*child_count=*/4, /*depth=*/6);
            tbb::task::enqueue(t);
        }
    }
    dummy_root->wait_for_all();
    tbb::task::destroy( *dummy_root );
}

class TaskList: public tbb::task {
    const int my_num_childs;
public:
    TaskList(const int num_childs) : my_num_childs(num_childs) {}
    tbb::task* execute() __TBB_override {
        tbb::task_list list;
        for (int i=0; i<my_num_childs; ++i)
        {
            list.push_back( *new( allocate_child() ) tbb::empty_task );
        }
        set_ref_count(my_num_childs+1);
        spawn(list);

        wait_for_all();
        return 0;
    }
};

void RunTaskListGenerator()
{
    const int max_num_childs = 10000;
    int num_childs=3;

    while ( num_childs < max_num_childs )
    {
        tbb::task& root = *new( tbb::task::allocate_root() ) TaskList(num_childs);

        tbb::task::spawn_root_and_wait(root);

        num_childs = 3 * num_childs;
    }
}

//! Tests whether task scheduler allows thieves to hoard task objects.
/** The test takes a while to run, so we run it only with the default
    number of threads. */
void TestTaskReclamation() {
    REMARK("testing task reclamation\n");

    size_t initial_amount_of_memory = 0;
    double task_count_sum = 0;
    double task_count_sum_square = 0;
    double average, sigma;

    tbb::task_scheduler_init init (MinThread);
    REMARK("Starting with %d threads\n", MinThread);
    // For now, the master will produce "additional" tasks; later a worker will replace it;
    Producer  = internal::governor::local_scheduler();
    int N = InitialStatsIterations;
    // First N iterations fill internal buffers and collect initial statistics
    for( int i=0; i<N; ++i ) {
        // First N iterations fill internal buffers and collect initial statistics
        RunTaskGenerators();
        RunTaskListGenerator();

        size_t m = GetMemoryUsage();
        if( m-initial_amount_of_memory > 0)
            initial_amount_of_memory = m;

        intptr_t n = internal::governor::local_scheduler()->get_task_node_count( /*count_arena_workers=*/true );
        task_count_sum += n;
        task_count_sum_square += n*n;

        REMARK( "Consumed %ld bytes and %ld objects (iteration=%d)\n", long(m), long(n), i );
    }
    // Calculate statistical values
    average = task_count_sum / N;
    sigma   = sqrt( (task_count_sum_square - task_count_sum*task_count_sum/N)/N );
    REMARK("Average task count: %g, sigma: %g, sum: %g, square sum:%g \n", average, sigma, task_count_sum, task_count_sum_square);

    int     last_error_iteration = 0,
            producer_switch_iteration = 0,
            producer_switches = 0;
    bool    switchProducer = false,
            checkProducer = false;
    for( int i=0; i < MaxIterations; ++i ) {
        // These iterations check for excessive memory use and unreasonable task count
        RunTaskGenerators( switchProducer, checkProducer );
        RunTaskListGenerator();

        intptr_t n = internal::governor::local_scheduler()->get_task_node_count( /*count_arena_workers=*/true );
        size_t m = GetMemoryUsage();

        if( (m-initial_amount_of_memory > 0) && (n > average+4*sigma) ) {
            // Use 4*sigma interval (for normal distribution, 3*sigma contains ~99% of values).
            REMARK( "Warning: possible leak of up to %ld bytes; currently %ld cached task objects (iteration=%d)\n",
                    static_cast<unsigned long>(m-initial_amount_of_memory), long(n), i );
            last_error_iteration = i;
            initial_amount_of_memory = m;
        } else {
            REMARK( "Consumed %ld bytes and %ld objects (iteration=%d)\n", long(m), long(n), i );
        }
        if ( i == last_error_iteration + AsymptoticRange ) {
            if ( producer_switches++ == NumProducerSwitches )
                break;
            else {
                last_error_iteration = producer_switch_iteration = i;
                switchProducer = true;
            }
        }
        else {
            switchProducer = false;
            checkProducer = producer_switch_iteration && (i == producer_switch_iteration + ProducerCheckTimeout);
        }
    }
    ASSERT( last_error_iteration < MaxIterations - AsymptoticRange, "The amount of allocated tasks keeps growing. Leak is possible." );
}

int TestMain () {
    if( !GetMemoryUsage() ) {
        REMARK("GetMemoryUsage is not implemented for this platform\n");
        return Harness::Skipped;
    }
    TestTaskReclamation();
    return Harness::Done;
}
