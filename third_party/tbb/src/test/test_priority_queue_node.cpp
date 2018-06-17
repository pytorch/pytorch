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

// TO DO: Add overlapping put / receive tests

#include "harness.h"
#include "tbb/flow_graph.h"
#include "harness_checktype.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "harness_graph.h"

#include <cstdio>

#define N 10
#define C 10

template< typename T >
void spin_try_get( tbb::flow::priority_queue_node<T> &q, T &value ) {
    while ( q.try_get(value) != true ) ;
}

template< typename T >
void check_item( T* next_value, T &value ) {
    int tid = value / N;
    int offset = value % N;
    ASSERT( next_value[tid] == T(offset), NULL );
    ++next_value[tid];
}

template< typename T >
struct parallel_puts : NoAssign {
    tbb::flow::priority_queue_node<T> &my_q;
    parallel_puts( tbb::flow::priority_queue_node<T> &q ) : my_q(q) {}
    void operator()(int i) const {
        for (int j = 0; j < N; ++j) {
            bool msg = my_q.try_put( T(N*i + j) );
            ASSERT( msg == true, NULL );
        }
    }
};

template< typename T >
struct parallel_gets : NoAssign {
    tbb::flow::priority_queue_node<T> &my_q;
    parallel_gets( tbb::flow::priority_queue_node<T> &q) : my_q(q) {}
    void operator()(int) const {
        T prev;
        spin_try_get( my_q, prev );
        for (int j = 0; j < N-1; ++j) {
            T v;
            spin_try_get( my_q, v );
            ASSERT(v < prev, NULL);
        }
    }
};

template< typename T >
struct parallel_put_get : NoAssign {
    tbb::flow::priority_queue_node<T> &my_q;
    parallel_put_get( tbb::flow::priority_queue_node<T> &q ) : my_q(q) {}
    void operator()(int tid) const {
        for ( int i = 0; i < N; i+=C ) {
            int j_end = ( N < i + C ) ? N : i + C;
            // dump about C values into the Q
            for ( int j = i; j < j_end; ++j ) {
                ASSERT( my_q.try_put( T (N*tid + j ) ) == true, NULL );
            }
            // receive about C values from the Q
            for ( int j = i; j < j_end; ++j ) {
                T v;
                spin_try_get( my_q, v );
            }
        }
    }
};

//
// Tests
//
// Item can be reserved, released, consumed ( single serial receiver )
//
template< typename T >
int test_reservation(int) {
    tbb::flow::graph g;

    // Simple tests
    tbb::flow::priority_queue_node<T> q(g);

    {

        T bogus_value(-1);

        q.try_put(T(1));
        q.try_put(T(2));
        q.try_put(T(3));
        g.wait_for_all();

        T v=bogus_value, w=bogus_value;
        ASSERT( q.try_reserve(v) == true, NULL );
        ASSERT( v == T(3), NULL );
        ASSERT( q.try_release() == true, NULL );
        v = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_reserve(v) == true, NULL );
        ASSERT( v == T(3), NULL );
        ASSERT( q.try_consume() == true, NULL );
        v = bogus_value;
        g.wait_for_all();

        ASSERT( q.try_get(v) == true, NULL );
        ASSERT( v == T(2), NULL );
        v = bogus_value;
        g.wait_for_all();

        ASSERT( q.try_reserve(v) == true, NULL );
        ASSERT( v == T(1), NULL );
        ASSERT( q.try_reserve(w) == false, NULL );
        ASSERT( w == bogus_value, NULL );
        ASSERT( q.try_get(w) == false, NULL );
        ASSERT( w == bogus_value, NULL );
        ASSERT( q.try_release() == true, NULL );
        v = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_reserve(v) == true, NULL );
        ASSERT( v == T(1), NULL );
        ASSERT( q.try_consume() == true, NULL );
        v = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_get(v) == false, NULL );
    }
    return 0;
}

//
// Tests
//
// multilpe parallel senders, items in FIFO (relatively to sender) order
// multilpe parallel senders, multiple parallel receivers, items in FIFO order (relative to sender/receiver) and all items received
//   * overlapped puts / gets
//   * all puts finished before any getS
//
template< typename T >
int test_parallel(int num_threads) {
    tbb::flow::graph g;
    tbb::flow::priority_queue_node<T> q(g);
    tbb::flow::priority_queue_node<T> q2(g);
    tbb::flow::priority_queue_node<T> q3(g);
    T bogus_value(-1);
    T j = bogus_value;

    NativeParallelFor( num_threads, parallel_puts<T>(q) );
    for (int i = num_threads*N -1; i>=0; --i) {
        spin_try_get( q, j );
        ASSERT(j == i, NULL);
        j = bogus_value;
    }
    g.wait_for_all();
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    NativeParallelFor( num_threads, parallel_puts<T>(q) );
    g.wait_for_all();
    NativeParallelFor( num_threads, parallel_gets<T>(q) );
    g.wait_for_all();
    j = bogus_value;
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    NativeParallelFor( num_threads, parallel_put_get<T>(q) );
    g.wait_for_all();
    j = bogus_value;
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    tbb::flow::make_edge( q, q2 );
    tbb::flow::make_edge( q2, q3 );
    NativeParallelFor( num_threads, parallel_puts<T>(q) );
    g.wait_for_all();
    NativeParallelFor( num_threads, parallel_gets<T>(q3) );
    g.wait_for_all();
    j = bogus_value;
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    ASSERT( q2.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    ASSERT( q3.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    // test copy constructor
    ASSERT( q.remove_successor( q2 ) == true, NULL );
    NativeParallelFor( num_threads, parallel_puts<T>(q) );
    tbb::flow::priority_queue_node<T> q_copy(q);
    g.wait_for_all();
    j = bogus_value;
    ASSERT( q_copy.try_get( j ) == false, NULL );
    ASSERT( q.register_successor( q_copy ) == true, NULL );
    for (int i = num_threads*N -1; i>=0; --i) {
        spin_try_get( q_copy, j );
        ASSERT(j == i, NULL);
        j = bogus_value;
    }
    g.wait_for_all();
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    ASSERT( q_copy.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    return 0;
}

//
// Tests
//
// Predecessors cannot be registered
// Empty Q rejects item requests
// Single serial sender, items in FIFO order
// Chained Qs ( 2 & 3 ), single sender, items at last Q in FIFO order
//

template< typename T >
int test_serial() {
    tbb::flow::graph g;
    T bogus_value(-1);

    tbb::flow::priority_queue_node<T> q(g);
    tbb::flow::priority_queue_node<T> q2(g);
    T j = bogus_value;

    //
    // Rejects attempts to add / remove predecessor
    // Rejects request from empty Q
    //
    ASSERT( q.register_predecessor( q2 ) == false, NULL );
    ASSERT( q.remove_predecessor( q2 ) == false, NULL );
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    //
    // Simple puts and gets
    //

    for (int i = 0; i < N; ++i)
        ASSERT( q.try_put( T(i) ), NULL );
    for (int i = N-1; i >=0; --i) {
        j = bogus_value;
        spin_try_get( q, j );
        ASSERT( i == j, NULL );
    }
    j = bogus_value;
    g.wait_for_all();
    ASSERT( q.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    tbb::flow::make_edge( q, q2 );

    for (int i = 0; i < N; ++i)
        ASSERT( q.try_put( T(i) ), NULL );
    g.wait_for_all();
    for (int i = N-1; i >= 0; --i) {
        j = bogus_value;
        spin_try_get( q2, j );
        ASSERT( i == j, NULL );
    }
    j = bogus_value;
    g.wait_for_all();
    ASSERT( q.try_get( j ) == false, NULL );
    g.wait_for_all();
    ASSERT( q2.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    tbb::flow::remove_edge( q, q2 );
    ASSERT( q.try_put( 1 ) == true, NULL );
    g.wait_for_all();
    ASSERT( q2.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    g.wait_for_all();
    ASSERT( q.try_get( j ) == true, NULL );
    ASSERT( j == 1, NULL );

    tbb::flow::priority_queue_node<T> q3(g);
    tbb::flow::make_edge( q, q2 );
    tbb::flow::make_edge( q2, q3 );

    for (int i = 0; i < N; ++i)
        ASSERT(  q.try_put( T(i) ), NULL );
    g.wait_for_all();
    for (int i = N-1; i >= 0; --i) {
        j = bogus_value;
        spin_try_get( q3, j );
        ASSERT( i == j, NULL );
    }
    j = bogus_value;
    g.wait_for_all();
    ASSERT( q.try_get( j ) == false, NULL );
    g.wait_for_all();
    ASSERT( q2.try_get( j ) == false, NULL );
    g.wait_for_all();
    ASSERT( q3.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );

    tbb::flow::remove_edge( q,  q2 );
    ASSERT( q.try_put( 1 ) == true, NULL );
    g.wait_for_all();
    ASSERT( q2.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    g.wait_for_all();
    ASSERT( q3.try_get( j ) == false, NULL );
    ASSERT( j == bogus_value, NULL );
    g.wait_for_all();
    ASSERT( q.try_get( j ) == true, NULL );
    ASSERT( j == 1, NULL );

    return 0;
}

int TestMain() {
    tbb::tick_count start = tbb::tick_count::now(), stop;
    for (int p = 2; p <= 4; ++p) {
        tbb::task_scheduler_init init(p);
        test_serial<int>();
        test_reservation<int>(p);
        test_reservation<check_type<int> >(p);
        test_parallel<int>(p);
    }
    stop = tbb::tick_count::now();
    REMARK("Priority_Queue_Node Time=%6.6f\n", (stop-start).seconds());
    REMARK("Testing resets\n");
    test_resets<int,tbb::flow::priority_queue_node<int> >();
    test_resets<float,tbb::flow::priority_queue_node<float> >();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    test_buffer_extract<tbb::flow::priority_queue_node<int> >().run_tests();
#endif
    return Harness::Done;
}
