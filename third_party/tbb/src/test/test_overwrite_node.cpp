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

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
#define TBB_PREVIEW_RESERVABLE_OVERWRITE_NODE 1
#endif

#include "harness_graph.h"

#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"

#define N 300
#define T 4
#define M 5

template< typename R >
void simple_read_write_tests() {
    tbb::flow::graph g;
    tbb::flow::overwrite_node<R> n(g);

    for ( int t = 0; t < T; ++t ) {
        R v0(N+1);
        std::vector< harness_counting_receiver<R> > r(M, harness_counting_receiver<R>(g));

        ASSERT( n.is_valid() == false, NULL );
        ASSERT( n.try_get( v0 ) == false, NULL );
        if ( t % 2 ) {
            ASSERT( n.try_put( static_cast<R>(N) ), NULL );
            ASSERT( n.is_valid() == true, NULL );
            ASSERT( n.try_get( v0 ) == true, NULL );
            ASSERT( v0 == R(N), NULL );
       }

        for (int i = 0; i < M; ++i) {
           tbb::flow::make_edge( n, r[i] );
        }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        ASSERT(n.successor_count() == M, NULL);
        typename tbb::flow::overwrite_node<R>::successor_list_type my_succs;
        n.copy_successors(my_succs);
        ASSERT(my_succs.size() == M, NULL);
        ASSERT(n.predecessor_count() == 0, NULL);
#endif

        for (int i = 0; i < N; ++i ) {
            R v1(static_cast<R>(i));
            ASSERT( n.try_put( v1 ), NULL );
            ASSERT( n.is_valid() == true, NULL );
            for (int j = 0; j < N; ++j ) {
                R v2(0);
                ASSERT( n.try_get( v2 ), NULL );
                ASSERT( v1 == v2, NULL );
            }
        }
        for (int i = 0; i < M; ++i) {
             size_t c = r[i].my_count;
             ASSERT( int(c) == N+t%2, NULL );
        }
        for (int i = 0; i < M; ++i) {
           tbb::flow::remove_edge( n, r[i] );
        }
        ASSERT( n.try_put( R(0) ), NULL );
        for (int i = 0; i < M; ++i) {
             size_t c = r[i].my_count;
             ASSERT( int(c) == N+t%2, NULL );
        }
        n.clear();
        ASSERT( n.is_valid() == false, NULL );
        ASSERT( n.try_get( v0 ) == false, NULL );
    }
}

template< typename R >
class native_body : NoAssign {
    tbb::flow::overwrite_node<R> &my_node;

public:

     native_body( tbb::flow::overwrite_node<R> &n ) : my_node(n) {}

     void operator()( int i ) const {
         R v1(static_cast<R>(i));
         ASSERT( my_node.try_put( v1 ), NULL );
         ASSERT( my_node.is_valid() == true, NULL );
     }
};

template< typename R >
void parallel_read_write_tests() {
    tbb::flow::graph g;
    tbb::flow::overwrite_node<R> n(g);
    //Create a vector of identical nodes
    std::vector< tbb::flow::overwrite_node<R> > ow_vec(2, n);

    for (size_t node_idx=0; node_idx<ow_vec.size(); ++node_idx) {
    for ( int t = 0; t < T; ++t ) {
        std::vector< harness_counting_receiver<R> > r(M, harness_counting_receiver<R>(g));

        for (int i = 0; i < M; ++i) {
           tbb::flow::make_edge( ow_vec[node_idx], r[i] );
        }
        R v0;
        ASSERT( ow_vec[node_idx].is_valid() == false, NULL );
        ASSERT( ow_vec[node_idx].try_get( v0 ) == false, NULL );

        NativeParallelFor( N, native_body<R>( ow_vec[node_idx] ) );

        for (int i = 0; i < M; ++i) {
             size_t c = r[i].my_count;
             ASSERT( int(c) == N, NULL );
        }
        for (int i = 0; i < M; ++i) {
           tbb::flow::remove_edge( ow_vec[node_idx], r[i] );
        }
        ASSERT( ow_vec[node_idx].try_put( R(0) ), NULL );
        for (int i = 0; i < M; ++i) {
             size_t c = r[i].my_count;
             ASSERT( int(c) == N, NULL );
        }
        ow_vec[node_idx].clear();
        ASSERT( ow_vec[node_idx].is_valid() == false, NULL );
        ASSERT( ow_vec[node_idx].try_get( v0 ) == false, NULL );
    }
    }
}

int TestMain() {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    simple_read_write_tests<int>();
    simple_read_write_tests<float>();
    for( int p=MinThread; p<=MaxThread; ++p ) {
        tbb::task_scheduler_init init(p);
        parallel_read_write_tests<int>();
        parallel_read_write_tests<float>();
#if TBB_PREVIEW_RESERVABLE_OVERWRITE_NODE
        test_reserving_nodes<tbb::flow::overwrite_node, int>();
#endif
    }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    test_extract_on_node<tbb::flow::overwrite_node, int>();
    test_extract_on_node<tbb::flow::overwrite_node, float>();
#endif
    return Harness::Done;
}

