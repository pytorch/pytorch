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

//TODO: when removing TBB_PREVIEW_LOCAL_OBSERVER, change the header or defines here
#include "tbb/task_scheduler_observer.h"

typedef uintptr_t FlagType;
const int MaxFlagIndex = sizeof(FlagType)*8-1;

class MyObserver: public tbb::task_scheduler_observer {
    FlagType flags;
    void on_scheduler_entry( bool is_worker ) __TBB_override;
    void on_scheduler_exit( bool is_worker ) __TBB_override;
public:
    MyObserver( FlagType flags_ ) : flags(flags_) {
        observe(true);
    }
};

#include "harness_assert.h"
#include "tbb/atomic.h"

tbb::atomic<int> EntryCount;
tbb::atomic<int> ExitCount;

struct State {
    FlagType MyFlags;
    bool IsMaster;
    State() : MyFlags(), IsMaster() {}
};

#include "../tbb/tls.h"
tbb::internal::tls<State*> LocalState;

void MyObserver::on_scheduler_entry( bool is_worker ) {
    State& state = *LocalState;
    ASSERT( is_worker==!state.IsMaster, NULL );
    ++EntryCount;
    state.MyFlags |= flags;
}

void MyObserver::on_scheduler_exit( bool is_worker ) {
    State& state = *LocalState;
    ASSERT( is_worker==!state.IsMaster, NULL );
    ++ExitCount;
    state.MyFlags &= ~flags;
}

#include "tbb/task.h"

class FibTask: public tbb::task {
    const int n;
    FlagType flags;
public:
    FibTask( int n_, FlagType flags_ ) : n(n_), flags(flags_) {}
    tbb::task* execute() __TBB_override {
        ASSERT( !(~LocalState->MyFlags & flags), NULL );
        if( n>=2 ) {
            set_ref_count(3);
            spawn(*new( allocate_child() ) FibTask(n-1,flags));
            spawn_and_wait_for_all(*new( allocate_child() ) FibTask(n-2,flags));
        }
        return NULL;
    }
};

void DoFib( FlagType flags ) {
    tbb::task* t = new( tbb::task::allocate_root() ) FibTask(10,flags);
    tbb::task::spawn_root_and_wait(*t);
}

#include "tbb/task_scheduler_init.h"
#include "harness.h"

class DoTest {
    int nthread;
public:
    DoTest( int n ) : nthread(n) {}
    void operator()( int i ) const {
        LocalState->IsMaster = true;
        if( i==0 ) {   
            tbb::task_scheduler_init init(nthread);
            DoFib(0);
        } else {
            FlagType f = i<=MaxFlagIndex? 1<<i : 0;
            MyObserver w(f);
            tbb::task_scheduler_init init(nthread);
            DoFib(f);
        }
    }
};

void TestObserver( int p, int q ) {
    NativeParallelFor( p, DoTest(q) );
}

int TestMain () {
    for( int p=MinThread; p<=MaxThread; ++p ) 
        for( int q=MinThread; q<=MaxThread; ++q ) 
            TestObserver(p,q);
    ASSERT( EntryCount>0, "on_scheduler_entry not exercised" );
    ASSERT( ExitCount>0, "on_scheduler_exit not exercised" );
    return Harness::Done;
}
