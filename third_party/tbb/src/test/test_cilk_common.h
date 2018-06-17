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

// This file is a common part of test_cilk_interop and test_cilk_dynamic_load tests

int TBB_Fib( int n );

class FibCilkSubtask: public tbb::task {
    int n;
    int& result;
    task* execute() __TBB_override {
        if( n<2 ) {
            result = n;
        } else {
            int x, y;
            x = cilk_spawn TBB_Fib(n-2);
            y = cilk_spawn TBB_Fib(n-1);
            cilk_sync;
            result = x+y;
        }
        return NULL;
    }
public:
    FibCilkSubtask( int& result_, int n_ ) : result(result_), n(n_) {}
};

class FibTask: public tbb::task {
    int n;
    int& result;
    task* execute() __TBB_override {
        if( !g_sandwich && n<2 ) {
            result = n;
        } else {
            int x,y;
            tbb::task_scheduler_init init(P_nested);
            task* self0 = &task::self();
            set_ref_count( 3 );
            if ( g_sandwich ) {
                spawn (*new( allocate_child() ) FibCilkSubtask(x,n-1));
                spawn (*new( allocate_child() ) FibCilkSubtask(y,n-2));
            }
            else {
                spawn (*new( allocate_child() ) FibTask(x,n-1));
                spawn (*new( allocate_child() ) FibTask(y,n-2));
            }
            wait_for_all();
            task* self1 = &task::self();
            ASSERT( self0 == self1, "failed to preserve TBB TLS" );
            result = x+y;
        }
        return NULL;
    }
public:
    FibTask( int& result_, int n_ ) : result(result_), n(n_) {}
};

int TBB_Fib( int n ) {
    if( n<2 ) {
        return n;
    } else {
        int result;
        tbb::task_scheduler_init init(P_nested);
        tbb::task::spawn_root_and_wait(*new( tbb::task::allocate_root()) FibTask(result,n) );
        return result;
    }
}
