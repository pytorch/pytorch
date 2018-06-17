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

#include "perf.h"

#include <cmath>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#define NUM_CHILD_TASKS     2096
#define NUM_ROOT_TASKS      256

#define N               100000000
#define FINEST_GRAIN    10
#define FINE_GRAIN      50
#define MED_GRAIN       200
#define COARSE_GRAIN    1000


typedef int count_t;

const count_t N_finest = (count_t)(N/log((double)N)/10);
const count_t N_fine = N_finest * 20;
const count_t N_med = N_fine * (count_t)log((double)N) / 5;

class StaticTaskHolder {
public:
    tbb::task *my_leafTaskPtr;
    StaticTaskHolder ();
};

static StaticTaskHolder s_tasks;

static count_t NumIterations;
static count_t NumLeafTasks;
static count_t NumRootTasks;

class LeafTaskBase : public tbb::task {
public:
    count_t my_ID;

    LeafTaskBase () {}
    LeafTaskBase ( count_t id ) : my_ID(id) {}
};

class SimpleLeafTask : public LeafTaskBase {
    task* execute () {
        volatile count_t anchor = 0;
        for ( count_t i=0; i < NumIterations; ++i )
            anchor += i;
        return NULL;
    }
public:
    SimpleLeafTask ( count_t ) {}
};

StaticTaskHolder::StaticTaskHolder () {
    static SimpleLeafTask s_t1(0);
    my_leafTaskPtr = &s_t1;
}

class Test_SPMC : public Perf::Test {
protected:
    static const int numWorkloads = 4;
    static const count_t workloads[numWorkloads];

    LeafTaskBase* my_leafTaskPtr;

    const char* Name () { return "SPMC"; }

    int NumWorkloads () { return numWorkloads; }

    void SetWorkload ( int idx ) {
        NumRootTasks = 1;
        NumIterations = workloads[idx];
        NumLeafTasks = NUM_CHILD_TASKS * NUM_ROOT_TASKS / (NumIterations > 1000 ? 32 : 8);
        Perf::SetWorkloadName( "%dx%d", NumLeafTasks, NumIterations );
    }
    
    void Run ( ThreadInfo& ) {
        tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
        r.set_ref_count( NumLeafTasks + 1 );
        for ( count_t i = 0; i < NumLeafTasks; ++i )
            r.spawn( *new(r.allocate_child()) SimpleLeafTask(0) );
        r.wait_for_all();
        tbb::task::destroy(r);
    }

    void RunSerial ( ThreadInfo& ) {
        const count_t n = NumLeafTasks * NumRootTasks;
        for ( count_t i=0; i < n; ++i ) {
            my_leafTaskPtr->my_ID = i;
            my_leafTaskPtr->execute();
        }
    }

public:
    Test_SPMC ( LeafTaskBase* leafTaskPtr = NULL ) {
        static SimpleLeafTask t(0);
        my_leafTaskPtr = leafTaskPtr ? leafTaskPtr : &t;
    }
}; // class Test_SPMC

const count_t Test_SPMC::workloads[Test_SPMC::numWorkloads] = { 1, 50, 500, 5000 };

template<class LeafTask>
class LeavesLauncherTask : public tbb::task {
    count_t my_groupId;

    task* execute () {
        count_t base = my_groupId * NumLeafTasks;
        set_ref_count(NumLeafTasks + 1);
        for ( count_t i = 0; i < NumLeafTasks; ++i )
            spawn( *new(allocate_child()) LeafTask(base + i) );
        wait_for_all();
        return NULL;
    }
public:
    LeavesLauncherTask ( count_t groupId ) : my_groupId(groupId) {}
};

template<class LeafTask>
void RunShallowTree () {
    tbb::empty_task &r = *new( tbb::task::allocate_root() ) tbb::empty_task;
    r.set_ref_count( NumRootTasks + 1 );
    for ( count_t i = 0; i < NumRootTasks; ++i )
        r.spawn( *new(r.allocate_child()) LeavesLauncherTask<LeafTask>(i) );
    r.wait_for_all();
    tbb::task::destroy(r);
}

class Test_ShallowTree : public Test_SPMC {
    const char* Name () { return "ShallowTree"; }

    void SetWorkload ( int idx ) {
        NumRootTasks = NUM_ROOT_TASKS;
        NumIterations = workloads[idx];
        NumLeafTasks = NumIterations > 200 ? NUM_CHILD_TASKS / 10 : 
                            (NumIterations > 50 ? NUM_CHILD_TASKS / 2 : NUM_CHILD_TASKS * 2);
        Perf::SetWorkloadName( "%dx%d", NumRootTasks * NumLeafTasks, NumIterations );
    }

    void Run ( ThreadInfo& ) {
        RunShallowTree<SimpleLeafTask>();
    }
}; // class Test_ShallowTree

class LeafTaskSkewed : public LeafTaskBase {
    task* execute () {
        volatile count_t anchor = 0;
        double K = (double)NumRootTasks * NumLeafTasks;
        count_t n = count_t(sqrt(double(my_ID)) * double(my_ID) * my_ID / (4 * K * K));
        for ( count_t i = 0; i < n; ++i )
            anchor += i;
        return NULL;
    }
public:
    LeafTaskSkewed ( count_t id ) : LeafTaskBase(id) {}
};

class Test_ShallowTree_Skewed : public Test_SPMC {
    static LeafTaskSkewed SerialTaskBody;

    const char* Name () { return "ShallowTree_Skewed"; }

    int NumWorkloads () { return 1; }

    void SetWorkload ( int ) {
        NumRootTasks = NUM_ROOT_TASKS;
        NumLeafTasks = NUM_CHILD_TASKS;
        Perf::SetWorkloadName( "%d", NumRootTasks * NumLeafTasks );
    }

    void Run ( ThreadInfo& ) {
        RunShallowTree<LeafTaskSkewed>();
    }

public:
    Test_ShallowTree_Skewed () : Test_SPMC(&SerialTaskBody) {}
}; // class Test_ShallowTree_Skewed

LeafTaskSkewed Test_ShallowTree_Skewed::SerialTaskBody(0);

typedef tbb::blocked_range<count_t> range_t;

static count_t  IterRange = N,
                IterGrain = 1;

enum PartitionerType {
    SimplePartitioner = 0,
    AutoPartitioner = 1
};

class Test_Algs : public Perf::Test {
protected:
    static const int numWorkloads = 4;
    static const count_t algRanges[numWorkloads];
    static const count_t algGrains[numWorkloads];

    tbb::simple_partitioner    my_simplePartitioner;
    tbb::auto_partitioner    my_autoPartitioner;
    PartitionerType my_partitionerType;

    bool UseAutoPartitioner () const { return my_partitionerType == AutoPartitioner; }

    int NumWorkloads () { return UseAutoPartitioner() ? 3 : numWorkloads; }

    void SetWorkload ( int idx ) {
        if ( UseAutoPartitioner() ) {
            IterRange = algRanges[idx ? numWorkloads - 1 : 0];
            IterGrain = idx > 1 ? algGrains[numWorkloads - 1] : 1;
        }
        else {
            IterRange = algRanges[idx];
            IterGrain = algGrains[idx];
        }
        Perf::SetWorkloadName( "%d/%d", IterRange, IterGrain );
    }
public:
    Test_Algs ( PartitionerType pt = SimplePartitioner ) : my_partitionerType(pt) {}
}; // class Test_Algs

const count_t Test_Algs::algRanges[] = {N_finest, N_fine, N_med, N};
const count_t Test_Algs::algGrains[] = {1, FINE_GRAIN, MED_GRAIN, COARSE_GRAIN};

template <typename Body>
class Test_PFor : public Test_Algs {
protected:
    void Run ( ThreadInfo& ) {
        if ( UseAutoPartitioner() )
            tbb::parallel_for( range_t(0, IterRange, IterGrain), Body(), my_autoPartitioner );
        else
            tbb::parallel_for( range_t(0, IterRange, IterGrain), Body(), my_simplePartitioner );
    }

    void RunSerial ( ThreadInfo& ) {
        Body body;
        body( range_t(0, IterRange, IterGrain) );
    }
public:
    Test_PFor ( PartitionerType pt = SimplePartitioner ) : Test_Algs(pt) {}
}; // class Test_PFor

class SimpleForBody {
public:
    void operator()( const range_t& r ) const {
        count_t end = r.end();
        volatile count_t anchor = 0;
        for( count_t i = r.begin(); i < end; ++i )
            anchor += i;
    }
}; // class SimpleForBody

class Test_PFor_Simple : public Test_PFor<SimpleForBody> {
protected:
    const char* Name () { return UseAutoPartitioner() ? "PFor-AP" : "PFor"; }
public:
    Test_PFor_Simple ( PartitionerType pt = SimplePartitioner ) : Test_PFor<SimpleForBody>(pt) {}
}; // class Test_PFor_Simple

class SkewedForBody {
public:
    void operator()( const range_t& r ) const {
        count_t end = (r.end() + 1) * (r.end() + 1);
        volatile count_t anchor = 0;
        for( count_t i = r.begin() * r.begin(); i < end; ++i )
            anchor += i;
    }
}; // class SkewedForBody

class Test_PFor_Skewed : public Test_PFor<SkewedForBody> {
    typedef Test_PFor<SkewedForBody> base_type;
protected:
    const char* Name () { return UseAutoPartitioner() ? "PFor-Skewed-AP" : "PFor-Skewed"; }

    void SetWorkload ( int idx ) {
        base_type::SetWorkload(idx);
        IterRange = (count_t)(sqrt((double)IterRange) * sqrt(sqrt((double)N / IterRange)));
        Perf::SetWorkloadName( "%d", IterRange );
    }

public:
    Test_PFor_Skewed ( PartitionerType pt = SimplePartitioner ) : base_type(pt) {}
}; // class Test_PFor_Skewed

PartitionerType gPartitionerType;
count_t NestingRange;
count_t NestingGrain;

class NestingForBody {
    count_t my_depth;
    tbb::simple_partitioner my_simplePartitioner;
    tbb::auto_partitioner my_autoPartitioner;
    
    template<class Partitioner>
    void run ( const range_t& r, Partitioner& p ) const {
        count_t end = r.end();
        if ( my_depth > 1 )
            for ( count_t i = r.begin(); i < end; ++i )
                tbb::parallel_for( range_t(0, IterRange, IterGrain), NestingForBody(my_depth - 1), p );
        else
            for ( count_t i = r.begin(); i < end; ++i )
                tbb::parallel_for( range_t(0, IterRange, IterGrain), SimpleForBody(), p );
    }
public:
    void operator()( const range_t& r ) const {
        if ( gPartitionerType == AutoPartitioner )
            run( r, my_autoPartitioner );
        else
            run( r, my_simplePartitioner );
    }
    NestingForBody ( count_t depth = 1 ) : my_depth(depth) {}
}; // class NestingForBody

enum NestingType {
    HollowNesting,
    ShallowNesting,
    DeepNesting
};

class Test_PFor_Nested : public Test_Algs {
    typedef Test_Algs base_type;

    NestingType my_nestingType;
    count_t my_nestingDepth;

protected:
    const char* Name () {
        static const char* names[] = { "PFor-HollowNested", "PFor-HollowNested-AP",
                                       "PFor-ShallowNested", "PFor-ShallowNested-AP",
                                       "PFor-DeeplyNested", "PFor-DeeplyNested-AP" };
        return names[my_nestingType * 2 + my_partitionerType];
    }

    int NumWorkloads () { return my_nestingType == ShallowNesting ? (UseAutoPartitioner() ? 3 : 2) : 1; }

    void SetWorkload ( int idx ) {
        gPartitionerType = my_partitionerType;
        if ( my_nestingType == DeepNesting ) {
            NestingRange = 1024;
            IterGrain = NestingGrain = 1;
            IterRange = 4;
            my_nestingDepth = 4;
        }
        else if ( my_nestingType == ShallowNesting ) {
            int i = idx ? numWorkloads - 1 : 0;
            count_t baseRange = algRanges[i];
            count_t baseGrain = !UseAutoPartitioner() || idx > 1 ? algGrains[i] : 1;
            NestingRange = IterRange = (count_t)sqrt((double)baseRange);
            NestingGrain = IterGrain = (count_t)sqrt((double)baseGrain);
        }
        else {
            NestingRange = N / 100;
            NestingGrain = COARSE_GRAIN / 10;
            IterRange = 2;
            IterGrain = 1;
        }
        Perf::SetWorkloadName( "%d/%d", NestingRange, NestingGrain );
    }

    void Run ( ThreadInfo& ) {
        if ( UseAutoPartitioner() )
            tbb::parallel_for( range_t(0, NestingRange, NestingGrain), NestingForBody(my_nestingDepth), my_autoPartitioner );
        else
            tbb::parallel_for( range_t(0, NestingRange, NestingGrain), NestingForBody(my_nestingDepth), my_simplePartitioner );
    }

    void RunSerial ( ThreadInfo& ) {
        for ( int i = 0; i < NestingRange; ++i ) {
            SimpleForBody body;
            body( range_t(0, IterRange, IterGrain) );
        }
    }
public:
    Test_PFor_Nested ( NestingType nt, PartitionerType pt ) : base_type(pt), my_nestingType(nt), my_nestingDepth(1) {}
}; // class Test_PFor_Nested

class SimpleReduceBody {
public:
    count_t my_sum;
    SimpleReduceBody () : my_sum(0) {}
    SimpleReduceBody ( SimpleReduceBody&, tbb::split ) : my_sum(0) {}
    void join( SimpleReduceBody& rhs ) { my_sum += rhs.my_sum;}
    void operator()( const range_t& r ) {
        count_t end = r.end();
        volatile count_t anchor = 0;
        for( count_t i = r.begin(); i < end; ++i )
            anchor += i;
        my_sum = anchor;
    }
}; // class SimpleReduceBody

class Test_PReduce : public Test_Algs {
protected:
    const char* Name () { return UseAutoPartitioner() ? "PReduce-AP" : "PReduce"; }

    void Run ( ThreadInfo& ) {
        SimpleReduceBody body;
        if ( UseAutoPartitioner() )
            tbb::parallel_reduce( range_t(0, IterRange, IterGrain), body, my_autoPartitioner );
        else
            tbb::parallel_reduce( range_t(0, IterRange, IterGrain), body, my_simplePartitioner );
    }

    void RunSerial ( ThreadInfo& ) {
        SimpleReduceBody body;
        body( range_t(0, IterRange, IterGrain) );
    }
public:
    Test_PReduce ( PartitionerType pt = SimplePartitioner ) : Test_Algs(pt) {}
}; // class Test_PReduce

int main( int argc, char* argv[] ) {
    Perf::SessionSettings opts (Perf::UseTaskScheduler | Perf::UseSerialBaseline, "perf_sched.txt");   // Perf::UseBaseline, Perf::UseSmallestWorkloadOnly
    Perf::RegisterTest<Test_SPMC>();
    Perf::RegisterTest<Test_ShallowTree>();
    Perf::RegisterTest<Test_ShallowTree_Skewed>();
    Test_PFor_Simple pf_sp(SimplePartitioner), pf_ap(AutoPartitioner);
    Perf::RegisterTest(pf_sp);
    Perf::RegisterTest(pf_ap);
    Test_PReduce pr_sp(SimplePartitioner), pr_ap(AutoPartitioner);
    Perf::RegisterTest(pr_sp);
    Perf::RegisterTest(pr_ap);
    Test_PFor_Skewed pf_s_sp(SimplePartitioner), pf_s_ap(AutoPartitioner);
    Perf::RegisterTest(pf_s_sp);
    Perf::RegisterTest(pf_s_ap);
    Test_PFor_Nested pf_hn_sp(HollowNesting, SimplePartitioner), pf_hn_ap(HollowNesting, AutoPartitioner),
                     pf_sn_sp(ShallowNesting, SimplePartitioner), pf_sn_ap(ShallowNesting, AutoPartitioner),
                     pf_dn_sp(DeepNesting, SimplePartitioner), pf_dn_ap(DeepNesting, AutoPartitioner);
    Perf::RegisterTest(pf_hn_sp);
    Perf::RegisterTest(pf_hn_ap);
    Perf::RegisterTest(pf_sn_sp);
    Perf::RegisterTest(pf_sn_ap);
    Perf::RegisterTest(pf_dn_sp);
    Perf::RegisterTest(pf_dn_ap);
    return Perf::TestMain(argc, argv, &opts);
}
