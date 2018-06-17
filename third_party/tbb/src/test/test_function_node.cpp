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

#include "harness_graph.h"

#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/spin_rw_mutex.h"

#define N 100
#define MAX_NODES 4

//! Performs test on function nodes with limited concurrency and buffering
/** Theses tests check:
    1) that the number of executing copies never exceed the concurrency limit
    2) that the node never rejects
    3) that no items are lost
    and 4) all of this happens even if there are multiple predecessors and successors
*/

template< typename InputType >
struct parallel_put_until_limit : private NoAssign {

    harness_counting_sender<InputType> *my_senders;

    parallel_put_until_limit( harness_counting_sender<InputType> *senders ) : my_senders(senders) {}

    void operator()( int i ) const  {
        if ( my_senders ) {
            my_senders[i].try_put_until_limit();
        }
    }

};

template<typename IO>
struct pass_through {
    IO operator()(const IO& i) { return i; }
};

template< typename InputType, typename OutputType, typename Body >
void buffered_levels( size_t concurrency, Body body ) {

   // Do for lc = 1 to concurrency level
   for ( size_t lc = 1; lc <= concurrency; ++lc ) {
   tbb::flow::graph g;

   // Set the execute_counter back to zero in the harness
   harness_graph_executor<InputType, OutputType>::execute_count = 0;
   // Set the number of current executors to zero.
   harness_graph_executor<InputType, OutputType>::current_executors = 0;
   // Set the max allowed executors to lc.  There is a check in the functor to make sure this is never exceeded.
   harness_graph_executor<InputType, OutputType>::max_executors = lc;

   // Create the function_node with the appropriate concurrency level, and use default buffering
   tbb::flow::function_node< InputType, OutputType > exe_node( g, lc, body );
   tbb::flow::function_node<InputType, InputType> pass_thru( g, tbb::flow::unlimited, pass_through<InputType>());

   // Create a vector of identical exe_nodes and pass_thrus
   std::vector< tbb::flow::function_node< InputType, OutputType > > exe_vec(2, exe_node);
   std::vector< tbb::flow::function_node< InputType, InputType > > pass_thru_vec(2, pass_thru);
   // Attach each pass_thru to its corresponding exe_node
   for (size_t node_idx=0; node_idx<exe_vec.size(); ++node_idx) {
       tbb::flow::make_edge(pass_thru_vec[node_idx], exe_vec[node_idx]);
   }

   // TODO: why the test is executed serially for the node pairs, not concurrently?
   for (size_t node_idx=0; node_idx<exe_vec.size(); ++node_idx) {
   // For num_receivers = 1 to MAX_NODES
   for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
        // Create num_receivers counting receivers and connect the exe_vec[node_idx] to them.
        std::vector< harness_mapped_receiver<OutputType>* > receivers(num_receivers);
        for (size_t i = 0; i < num_receivers; i++) {
            receivers[i] = new harness_mapped_receiver<OutputType>(g);
        }

        for (size_t r = 0; r < num_receivers; ++r ) {
            tbb::flow::make_edge( exe_vec[node_idx], *receivers[r] );
        }

        // Do the test with varying numbers of senders
        harness_counting_sender<InputType> *senders = NULL;
        for (size_t num_senders = 1; num_senders <= MAX_NODES; ++num_senders ) {
            // Create num_senders senders, set there message limit each to N, and connect them to pass_thru_vec[node_idx]
            senders = new harness_counting_sender<InputType>[num_senders];
            for (size_t s = 0; s < num_senders; ++s ) {
               senders[s].my_limit = N;
               senders[s].register_successor(pass_thru_vec[node_idx] );
            }

            // Initialize the receivers so they know how many senders and messages to check for
            for (size_t r = 0; r < num_receivers; ++r ) {
                 receivers[r]->initialize_map( N, num_senders );
            }

            // Do the test
            NativeParallelFor( (int)num_senders, parallel_put_until_limit<InputType>(senders) );
            g.wait_for_all();

            // confirm that each sender was requested from N times
            for (size_t s = 0; s < num_senders; ++s ) {
                size_t n = senders[s].my_received;
                ASSERT( n == N, NULL );
                ASSERT( senders[s].my_receiver == &pass_thru_vec[node_idx], NULL );
            }
            // validate the receivers
            for (size_t r = 0; r < num_receivers; ++r ) {
                receivers[r]->validate();
            }
            delete [] senders;
        }
        for (size_t r = 0; r < num_receivers; ++r ) {
            tbb::flow::remove_edge( exe_vec[node_idx], *receivers[r] );
        }
        ASSERT( exe_vec[node_idx].try_put( InputType() ) == true, NULL );
        g.wait_for_all();
        for (size_t r = 0; r < num_receivers; ++r ) {
            // since it's detached, nothing should have changed
            receivers[r]->validate();
        }

        for (size_t i = 0; i < num_receivers; i++) {
            delete receivers[i];
        }

    } // for num_receivers
    } // for node_idx
    } // for concurrency level lc
}

const size_t Offset = 123;
tbb::atomic<size_t> global_execute_count;

struct inc_functor {

    tbb::atomic<size_t> local_execute_count;
    inc_functor( ) { local_execute_count = 0; }
    inc_functor( const inc_functor &f ) { local_execute_count = f.local_execute_count; }
    void operator=( const inc_functor &f ) { local_execute_count = f.local_execute_count; }

    int operator()( int i ) {
       ++global_execute_count;
       ++local_execute_count;
       return i;
    }

};

template< typename InputType, typename OutputType >
void buffered_levels_with_copy( size_t concurrency ) {

    // Do for lc = 1 to concurrency level
    for ( size_t lc = 1; lc <= concurrency; ++lc ) {
        tbb::flow::graph g;

        inc_functor cf;
        cf.local_execute_count = Offset;
        global_execute_count = Offset;

        tbb::flow::function_node< InputType, OutputType > exe_node( g, lc, cf );

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {

           std::vector< harness_mapped_receiver<OutputType>* > receivers(num_receivers);
           for (size_t i = 0; i < num_receivers; i++) {
               receivers[i] = new harness_mapped_receiver<OutputType>(g);
           }

           for (size_t r = 0; r < num_receivers; ++r ) {
               tbb::flow::make_edge( exe_node, *receivers[r] );
            }

            harness_counting_sender<InputType> *senders = NULL;
            for (size_t num_senders = 1; num_senders <= MAX_NODES; ++num_senders ) {
                senders = new harness_counting_sender<InputType>[num_senders];
                for (size_t s = 0; s < num_senders; ++s ) {
                    senders[s].my_limit = N;
                    tbb::flow::make_edge( senders[s], exe_node );
                }

                for (size_t r = 0; r < num_receivers; ++r ) {
                    receivers[r]->initialize_map( N, num_senders );
                }

                NativeParallelFor( (int)num_senders, parallel_put_until_limit<InputType>(senders) );
                g.wait_for_all();

                for (size_t s = 0; s < num_senders; ++s ) {
                    size_t n = senders[s].my_received;
                    ASSERT( n == N, NULL );
                    ASSERT( senders[s].my_receiver == &exe_node, NULL );
                }
                for (size_t r = 0; r < num_receivers; ++r ) {
                    receivers[r]->validate();
                }
                delete [] senders;
            }
            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::remove_edge( exe_node, *receivers[r] );
            }
            ASSERT( exe_node.try_put( InputType() ) == true, NULL );
            g.wait_for_all();
            for (size_t r = 0; r < num_receivers; ++r ) {
                receivers[r]->validate();
            }

            for (size_t i = 0; i < num_receivers; i++) {
                delete receivers[i];
            }
        }

        // validate that the local body matches the global execute_count and both are correct
        inc_functor body_copy = tbb::flow::copy_body<inc_functor>( exe_node );
        const size_t expected_count = N/2 * MAX_NODES * MAX_NODES * ( MAX_NODES + 1 ) + MAX_NODES + Offset;
        size_t global_count = global_execute_count;
        size_t inc_count = body_copy.local_execute_count;
        ASSERT( global_count == expected_count && global_count == inc_count, NULL );
        g.reset(tbb::flow::rf_reset_bodies);
        body_copy = tbb::flow::copy_body<inc_functor>( exe_node );
        inc_count = body_copy.local_execute_count;
        ASSERT( Offset == inc_count, "reset(rf_reset_bodies) did not reset functor" );
    }
}

template< typename InputType, typename OutputType >
void run_buffered_levels( int c ) {
    #if __TBB_CPP11_LAMBDAS_PRESENT
    buffered_levels<InputType,OutputType>( c, []( InputType i ) -> OutputType { return harness_graph_executor<InputType, OutputType>::func(i); } );
    #endif
    buffered_levels<InputType,OutputType>( c, &harness_graph_executor<InputType, OutputType>::func );
    buffered_levels<InputType,OutputType>( c, typename harness_graph_executor<InputType, OutputType>::functor() );
    buffered_levels_with_copy<InputType,OutputType>( c );
}


//! Performs test on executable nodes with limited concurrency
/** Theses tests check:
    1) that the nodes will accepts puts up to the concurrency limit,
    2) the nodes do not exceed the concurrency limit even when run with more threads (this is checked in the harness_graph_executor),
    3) the nodes will receive puts from multiple successors simultaneously,
    and 4) the nodes will send to multiple predecessors.
    There is no checking of the contents of the messages for corruption.
*/

template< typename InputType, typename OutputType, typename Body >
void concurrency_levels( size_t concurrency, Body body ) {

   for ( size_t lc = 1; lc <= concurrency; ++lc ) {
       tbb::flow::graph g;

       // Set the execute_counter back to zero in the harness
       harness_graph_executor<InputType, OutputType>::execute_count = 0;
       // Set the number of current executors to zero.
       harness_graph_executor<InputType, OutputType>::current_executors = 0;
       // Set the max allowed executors to lc. There is a check in the functor to make sure this is never exceeded.
       harness_graph_executor<InputType, OutputType>::max_executors = lc;

       typedef tbb::flow::function_node< InputType, OutputType, tbb::flow::rejecting > fnode_type;
       fnode_type exe_node( g, lc, body );

       for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {

            std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            ASSERT(exe_node.successor_count() == 0, NULL);
            ASSERT(exe_node.predecessor_count() == 0, NULL);
#endif

            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::make_edge( exe_node, receivers[r] );
            }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            ASSERT(exe_node.successor_count() == num_receivers, NULL);
            typename fnode_type::successor_list_type my_succs;
            exe_node.copy_successors(my_succs);
            ASSERT(my_succs.size() == num_receivers, NULL);
            typename fnode_type::predecessor_list_type my_preds;
            exe_node.copy_predecessors(my_preds);
            ASSERT(my_preds.size() == 0, NULL);
#endif

            harness_counting_sender<InputType> *senders = NULL;

            for (size_t num_senders = 1; num_senders <= MAX_NODES; ++num_senders ) {
                senders = new harness_counting_sender<InputType>[num_senders];
                {
                    // Exclusively lock m to prevent exe_node from finishing
                    tbb::spin_rw_mutex::scoped_lock l( harness_graph_executor<InputType, OutputType>::template mutex_holder<tbb::spin_rw_mutex>::mutex );

                    // put to lc level, it will accept and then block at m
                    for ( size_t c = 0 ; c < lc ; ++c ) {
                        ASSERT( exe_node.try_put( InputType() ) == true, NULL );
                    }
                    // it only accepts to lc level
                    ASSERT( exe_node.try_put( InputType() ) == false, NULL );

                    for (size_t s = 0; s < num_senders; ++s ) {
                       // register a sender
                       senders[s].my_limit = N;
                       exe_node.register_predecessor( senders[s] );
                    }

                } // release lock at end of scope, setting the exe node free to continue
                // wait for graph to settle down
                g.wait_for_all();

                // confirm that each sender was requested from N times
                for (size_t s = 0; s < num_senders; ++s ) {
                    size_t n = senders[s].my_received;
                    ASSERT( n == N, NULL );
                    ASSERT( senders[s].my_receiver == &exe_node, NULL );
                }
                // confirm that each receivers got N * num_senders + the initial lc puts
                for (size_t r = 0; r < num_receivers; ++r ) {
                    size_t n = receivers[r].my_count;
                    ASSERT( n == num_senders*N+lc, NULL );
                    receivers[r].my_count = 0;
                }
                delete [] senders;
            }
            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::remove_edge( exe_node, receivers[r] );
            }
            ASSERT( exe_node.try_put( InputType() ) == true, NULL );
            g.wait_for_all();
            for (size_t r = 0; r < num_receivers; ++r ) {
                ASSERT( int(receivers[r].my_count) == 0, NULL );
            }
        }
    }
}


template< typename InputType, typename OutputType >
void run_concurrency_levels( int c ) {
    #if __TBB_CPP11_LAMBDAS_PRESENT
    concurrency_levels<InputType,OutputType>( c, []( InputType i ) -> OutputType { return harness_graph_executor<InputType, OutputType>::template tfunc<tbb::spin_rw_mutex>(i); } );
    #endif
    concurrency_levels<InputType,OutputType>( c, &harness_graph_executor<InputType, OutputType>::template tfunc<tbb::spin_rw_mutex> );
    concurrency_levels<InputType,OutputType>( c, typename harness_graph_executor<InputType, OutputType>::template tfunctor<tbb::spin_rw_mutex>() );
}


struct empty_no_assign {
   empty_no_assign() {}
   empty_no_assign( int ) {}
   operator int() { return 0; }
};

template< typename InputType >
struct parallel_puts : private NoAssign {

    tbb::flow::receiver< InputType > * const my_exe_node;

    parallel_puts( tbb::flow::receiver< InputType > &exe_node ) : my_exe_node(&exe_node) {}

    void operator()( int ) const  {
        for ( int i = 0; i < N; ++i ) {
            // the nodes will accept all puts
            ASSERT( my_exe_node->try_put( InputType() ) == true, NULL );
        }
    }

};

//! Performs test on executable nodes with unlimited concurrency
/** Theses tests check:
    1) that the nodes will accept all puts
    2) the nodes will receive puts from multiple predecessors simultaneously,
    and 3) the nodes will send to multiple successors.
    There is no checking of the contents of the messages for corruption.
*/

template< typename InputType, typename OutputType, typename Body >
void unlimited_concurrency( Body body ) {

    for (int p = 1; p < 2*MaxThread; ++p) {
        tbb::flow::graph g;
        tbb::flow::function_node< InputType, OutputType, tbb::flow::rejecting > exe_node( g, tbb::flow::unlimited, body );

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {

            std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));
            harness_graph_executor<InputType, OutputType>::execute_count = 0;

            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::make_edge( exe_node, receivers[r] );
            }

            NativeParallelFor( p, parallel_puts<InputType>(exe_node) );
            g.wait_for_all();

            // 2) the nodes will receive puts from multiple predecessors simultaneously,
            size_t ec = harness_graph_executor<InputType, OutputType>::execute_count;
            ASSERT( (int)ec == p*N, NULL );
            for (size_t r = 0; r < num_receivers; ++r ) {
                size_t c = receivers[r].my_count;
                // 3) the nodes will send to multiple successors.
                ASSERT( (int)c == p*N, NULL );
            }
            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::remove_edge( exe_node, receivers[r] );
            }
            }
        }
    }

template< typename InputType, typename OutputType >
void run_unlimited_concurrency() {
    harness_graph_executor<InputType, OutputType>::max_executors = 0;
    #if __TBB_CPP11_LAMBDAS_PRESENT
    unlimited_concurrency<InputType,OutputType>( []( InputType i ) -> OutputType { return harness_graph_executor<InputType, OutputType>::func(i); } );
    #endif
    unlimited_concurrency<InputType,OutputType>( &harness_graph_executor<InputType, OutputType>::func );
    unlimited_concurrency<InputType,OutputType>( typename harness_graph_executor<InputType, OutputType>::functor() );
}

struct continue_msg_to_int {
    int my_int;
    continue_msg_to_int(int x) : my_int(x) {}
    int operator()(tbb::flow::continue_msg) { return my_int; }
};

void test_function_node_with_continue_msg_as_input() {
    // If this function terminates, then this test is successful
    tbb::flow::graph g;

    tbb::flow::broadcast_node<tbb::flow::continue_msg> Start(g);

    tbb::flow::function_node<tbb::flow::continue_msg, int, tbb::flow::rejecting> FN1( g, tbb::flow::serial, continue_msg_to_int(42));
    tbb::flow::function_node<tbb::flow::continue_msg, int, tbb::flow::rejecting> FN2( g, tbb::flow::serial, continue_msg_to_int(43));

    tbb::flow::make_edge( Start, FN1 );
    tbb::flow::make_edge( Start, FN2 );

    Start.try_put( tbb::flow::continue_msg() );
    g.wait_for_all();
}

//! Tests limited concurrency cases for nodes that accept data messages
void test_concurrency(int num_threads) {
    tbb::task_scheduler_init init(num_threads);
    run_concurrency_levels<int,int>(num_threads);
    run_concurrency_levels<int,tbb::flow::continue_msg>(num_threads);
    run_buffered_levels<int, int>(num_threads);
    run_unlimited_concurrency<int,int>();
    run_unlimited_concurrency<int,empty_no_assign>();
    run_unlimited_concurrency<empty_no_assign,int>();
    run_unlimited_concurrency<empty_no_assign,empty_no_assign>();
    run_unlimited_concurrency<int,tbb::flow::continue_msg>();
    run_unlimited_concurrency<empty_no_assign,tbb::flow::continue_msg>();
    test_function_node_with_continue_msg_as_input();
}

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
struct add_to_counter {
    int* counter;
    add_to_counter(int& var):counter(&var){}
    int operator()(int i){*counter+=1; return i + 1;}
};

template<typename FTYPE>
void test_extract() {
    int my_count = 0;
    int cm;
    tbb::flow::graph g;
    tbb::flow::broadcast_node<int> b0(g);
    tbb::flow::broadcast_node<int> b1(g);
    tbb::flow::function_node<int, int, FTYPE> f0(g, tbb::flow::unlimited, add_to_counter(my_count));
    tbb::flow::queue_node<int> q0(g);

    tbb::flow::make_edge(b0, f0);
    tbb::flow::make_edge(b1, f0);
    tbb::flow::make_edge(f0, q0);
    for( int i = 0; i < 2; ++i ) {
        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(f0.predecessor_count() == 2 && f0.successor_count() == 1, "f0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");

        /* b0         */
        /*   \        */
        /*    f0 - q0 */
        /*   /        */
        /* b1         */

        b0.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 1, "function_node didn't fire");
        ASSERT(q0.try_get(cm), "function_node didn't forward");
        b1.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 2, "function_node didn't fire");
        ASSERT(q0.try_get(cm), "function_node didn't forward");

        b0.extract();

        /* b0         */
        /*            */
        /*    f0 - q0 */
        /*   /        */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(f0.predecessor_count() == 1 && f0.successor_count() == 1, "f0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(1);
        b0.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 2, "b0 messages being forwarded to function_node even though it is disconnected");
        b1.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 3, "function_node didn't fire though it has only one predecessor");
        ASSERT(q0.try_get(cm), "function_node didn't forward second time");

        f0.extract();

        /* b0         */
        /*            */
        /*    f0   q0 */
        /*            */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(f0.predecessor_count() == 0 && f0.successor_count() == 0, "f0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(1);
        b0.try_put(1);
        b1.try_put(1);
        b1.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 3, "function_node didn't fire though it has only one predecessor");
        ASSERT(!q0.try_get(cm), "function_node forwarded though it shouldn't");
        make_edge(b0, f0);

        /* b0         */
        /*   \        */
        /*    f0   q0 */
        /*            */
        /* b1         */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(f0.predecessor_count() == 1 && f0.successor_count() == 0, "f0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");

        b0.try_put(int());
        g.wait_for_all();

        ASSERT(my_count == 4, "function_node didn't fire though it has only one predecessor");
        ASSERT(!q0.try_get(cm), "function_node forwarded though it shouldn't");

        tbb::flow::make_edge(b1, f0);
        tbb::flow::make_edge(f0, q0);
        my_count = 0;
    }
}
#endif

int TestMain() {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
       test_concurrency(p);
   }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    test_extract<tbb::flow::rejecting>();
    test_extract<tbb::flow::queueing>();
#endif
   return Harness::Done;
}

