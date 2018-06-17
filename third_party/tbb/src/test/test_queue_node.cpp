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
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"
#include "harness_checktype.h"
#include "harness_graph.h"

#include <cstdio>

#define N 1000
#define C 10

template< typename T >
void spin_try_get( tbb::flow::queue_node<T> &q, T &value ) {
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

    tbb::flow::queue_node<T> &my_q;

    parallel_puts( tbb::flow::queue_node<T> &q ) : my_q(q) {}

    void operator()(int i) const {
        for (int j = 0; j < N; ++j) {
            bool msg = my_q.try_put( T(N*i + j) );
            ASSERT( msg == true, NULL );
        }
    }

};



template< typename T >
struct touches {

    bool **my_touches;
    T **my_last_touch;
    int my_num_threads;

    touches( int num_threads ) : my_num_threads(num_threads) {
        my_last_touch = new T* [my_num_threads];
        my_touches = new bool* [my_num_threads];
        for ( int p = 0; p < my_num_threads; ++p) {
            my_last_touch[p] = new T[my_num_threads];
            for ( int p2 = 0; p2 < my_num_threads; ++p2)
                my_last_touch[p][p2] = -1;

            my_touches[p] = new bool[N*my_num_threads];
            for ( int n = 0; n < N*my_num_threads; ++n)
                my_touches[p][n] = false;
        }
    }

    ~touches() {
        for ( int p = 0; p < my_num_threads; ++p) {
            delete [] my_touches[p];
            delete [] my_last_touch[p];
        }
        delete [] my_touches;
        delete [] my_last_touch;
    }

    bool check( int tid, T v ) {
        int v_tid = v / N;
        if ( my_touches[tid][v] != false ) {
            printf("Error: value seen twice by local thread\n");
            return false;
        }
        if ( v <= my_last_touch[tid][v_tid] ) {
            printf("Error: value seen in wrong order by local thread\n");
            return false;
        }
        my_last_touch[tid][v_tid] = v;
        my_touches[tid][v] = true;
        return true;
    }

    bool validate_touches() {
        bool *all_touches = new bool[N*my_num_threads];
        for ( int n = 0; n < N*my_num_threads; ++n)
            all_touches[n] = false;

        for ( int p = 0; p < my_num_threads; ++p) {
            for ( int n = 0; n < N*my_num_threads; ++n) {
                if ( my_touches[p][n] == true ) {
                    ASSERT( all_touches[n] == false, "value see by more than one thread\n" );
                    all_touches[n] = true;
                }
            }
        }
        for ( int n = 0; n < N*my_num_threads; ++n) {
            if ( !all_touches[n] )
                printf("No touch at %d, my_num_threads = %d\n", n, my_num_threads);
            //ASSERT( all_touches[n] == true, "value not seen by any thread\n" );
        }
        delete [] all_touches;
        return true;
    }

};

template< typename T >
struct parallel_gets : NoAssign {

    tbb::flow::queue_node<T> &my_q;
    touches<T> &my_touches;

    parallel_gets( tbb::flow::queue_node<T> &q, touches<T> &t) : my_q(q), my_touches(t) {}

    void operator()(int tid) const {
        for (int j = 0; j < N; ++j) {
            T v;
            spin_try_get( my_q, v );
            my_touches.check( tid, v );
        }
    }

};

template< typename T >
struct parallel_put_get : NoAssign {

    tbb::flow::queue_node<T> &my_q;
    touches<T> &my_touches;

    parallel_put_get( tbb::flow::queue_node<T> &q, touches<T> &t ) : my_q(q), my_touches(t) {}

    void operator()(int tid) const {

        for ( int i = 0; i < N; i+=C ) {
            int j_end = ( N < i + C ) ? N : i + C;
            // dump about C values into the Q
            for ( int j = i; j < j_end; ++j ) {
                ASSERT( my_q.try_put( T (N*tid + j ) ) == true, NULL );
            }
            // receiver about C values from the Q
            for ( int j = i; j < j_end; ++j ) {
                T v;
                spin_try_get( my_q, v );
                my_touches.check( tid, v );
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
int test_reservation() {
    tbb::flow::graph g;
    T bogus_value(-1);

    // Simple tests
    tbb::flow::queue_node<T> q(g);

    q.try_put(T(1));
    q.try_put(T(2));
    q.try_put(T(3));

    T v;
    ASSERT( q.reserve_item(v) == true, NULL );
    ASSERT( v == T(1), NULL );
    ASSERT( q.release_reservation() == true, NULL );
    v = bogus_value;
    g.wait_for_all();
    ASSERT( q.reserve_item(v) == true, NULL );
    ASSERT( v == T(1), NULL );
    ASSERT( q.consume_reservation() == true, NULL );
    v = bogus_value;
    g.wait_for_all();

    ASSERT( q.try_get(v) == true, NULL );
    ASSERT( v == T(2), NULL );
    v = bogus_value;
    g.wait_for_all();

    ASSERT( q.reserve_item(v) == true, NULL );
    ASSERT( v == T(3), NULL );
    ASSERT( q.release_reservation() == true, NULL );
    v = bogus_value;
    g.wait_for_all();
    ASSERT( q.reserve_item(v) == true, NULL );
    ASSERT( v == T(3), NULL );
    ASSERT( q.consume_reservation() == true, NULL );
    v = bogus_value;
    g.wait_for_all();

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
    tbb::flow::queue_node<T> q(g);
    tbb::flow::queue_node<T> q2(g);
    tbb::flow::queue_node<T> q3(g);
    {
        Check< T > my_check;
        T bogus_value(-1);
        T j = bogus_value;
        NativeParallelFor( num_threads, parallel_puts<T>(q) );

        T *next_value = new T[num_threads];
        for (int tid = 0; tid < num_threads; ++tid) next_value[tid] = T(0);

        for (int i = 0; i < num_threads * N; ++i ) {
            spin_try_get( q, j );
            check_item( next_value, j );
            j = bogus_value;
        }
        for (int tid = 0; tid < num_threads; ++tid)  {
            ASSERT( next_value[tid] == T(N), NULL );
        }
        delete[] next_value;

        j = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );

        NativeParallelFor( num_threads, parallel_puts<T>(q) );

        {
            touches< T > t( num_threads );
            NativeParallelFor( num_threads, parallel_gets<T>(q, t) );
            g.wait_for_all();
            ASSERT( t.validate_touches(), NULL );
        }
        j = bogus_value;
        ASSERT( q.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );

        g.wait_for_all();
        {
            touches< T > t2( num_threads );
            NativeParallelFor( num_threads, parallel_put_get<T>(q, t2) );
            g.wait_for_all();
            ASSERT( t2.validate_touches(), NULL );
        }
        j = bogus_value;
        ASSERT( q.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );

        tbb::flow::make_edge( q, q2 );
        tbb::flow::make_edge( q2, q3 );

        NativeParallelFor( num_threads, parallel_puts<T>(q) );
        {
            touches< T > t3( num_threads );
            NativeParallelFor( num_threads, parallel_gets<T>(q3, t3) );
            g.wait_for_all();
            ASSERT( t3.validate_touches(), NULL );
        }
        j = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_get( j ) == false, NULL );
        g.wait_for_all();
        ASSERT( q2.try_get( j ) == false, NULL );
        g.wait_for_all();
        ASSERT( q3.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );

        // test copy constructor
        ASSERT( q.remove_successor( q2 ), NULL );
        NativeParallelFor( num_threads, parallel_puts<T>(q) );
        tbb::flow::queue_node<T> q_copy(q);
        j = bogus_value;
        g.wait_for_all();
        ASSERT( q_copy.try_get( j ) == false, NULL );
        ASSERT( q.register_successor( q_copy ) == true, NULL );
        {
            touches< T > t( num_threads );
            NativeParallelFor( num_threads, parallel_gets<T>(q_copy, t) );
            g.wait_for_all();
            ASSERT( t.validate_touches(), NULL );
        }
        j = bogus_value;
        ASSERT( q.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );
        ASSERT( q_copy.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );
    }

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
    tbb::flow::queue_node<T> q(g);
    tbb::flow::queue_node<T> q2(g);
    {   // destroy the graph after manipulating it, and see if all the items in the buffers
        // have been destroyed before the graph
        Check<T> my_check;  // if check_type< U > count constructions and destructions
        T bogus_value(-1);
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

        for (int i = 0; i < N; ++i) {
            bool msg = q.try_put( T(i) );
            ASSERT( msg == true, NULL );
        }


        for (int i = 0; i < N; ++i) {
            j = bogus_value;
            spin_try_get( q, j );
            ASSERT( i == j, NULL );
        }
        j = bogus_value;
        g.wait_for_all();
        ASSERT( q.try_get( j ) == false, NULL );
        ASSERT( j == bogus_value, NULL );

        tbb::flow::make_edge( q, q2 );

        for (int i = 0; i < N; ++i) {
            bool msg = q.try_put( T(i) );
            ASSERT( msg == true, NULL );
        }


        for (int i = 0; i < N; ++i) {
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

        tbb::flow::queue_node<T> q3(g);
        tbb::flow::make_edge( q, q2 );
        tbb::flow::make_edge( q2, q3 );

        for (int i = 0; i < N; ++i) {
            bool msg = q.try_put( T(i) );
            ASSERT( msg == true, NULL );
        }

        for (int i = 0; i < N; ++i) {
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
    }

    return 0;
}

int TestMain() {
    tbb::tick_count start = tbb::tick_count::now(), stop;
    for (int p = 2; p <= 4; ++p) {
        tbb::task_scheduler_init init(p);
        test_serial<int>();
        test_serial<check_type<int> >();
        test_parallel<int>(p);
        test_parallel<check_type<int> >(p);
    }
    stop = tbb::tick_count::now();
    REMARK("Queue_Node Time=%6.6f\n", (stop-start).seconds());
    REMARK("Testing resets\n");
    test_resets<int, tbb::flow::queue_node<int> >();
    test_resets<float, tbb::flow::queue_node<float> >();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    test_buffer_extract<tbb::flow::queue_node<int> >().run_tests();
#endif
    return Harness::Done;
}
