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

#if TBB_USE_DEBUG
#define N 16
#else
#define N 100
#endif
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

//! exercise buffered multifunction_node.
template< typename InputType, typename OutputTuple, typename Body >
void buffered_levels( size_t concurrency, Body body ) {
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type OutputType;
    // Do for lc = 1 to concurrency level
    for ( size_t lc = 1; lc <= concurrency; ++lc ) {
        tbb::flow::graph g;

        // Set the execute_counter back to zero in the harness
        harness_graph_multifunction_executor<InputType, OutputTuple>::execute_count = 0;
        // Set the number of current executors to zero.
        harness_graph_multifunction_executor<InputType, OutputTuple>::current_executors = 0;
        // Set the max allowed executors to lc.  There is a check in the functor to make sure this is never exceeded.
        harness_graph_multifunction_executor<InputType, OutputTuple>::max_executors = lc;

        // Create the function_node with the appropriate concurrency level, and use default buffering
        tbb::flow::multifunction_node< InputType, OutputTuple > exe_node( g, lc, body );

        //Create a vector of identical exe_nodes
        std::vector< tbb::flow::multifunction_node< InputType, OutputTuple > > exe_vec(2, exe_node);

        // exercise each of the copied nodes
        for (size_t node_idx=0; node_idx<exe_vec.size(); ++node_idx) {
            for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
                // Create num_receivers counting receivers and connect the exe_vec[node_idx] to them.
                std::vector< harness_mapped_receiver<OutputType>* > receivers(num_receivers);
                for (size_t i = 0; i < num_receivers; i++) {
                    receivers[i] = new harness_mapped_receiver<OutputType>(g);
                }

                for (size_t r = 0; r < num_receivers; ++r ) {
                    tbb::flow::make_edge( tbb::flow::output_port<0>(exe_vec[node_idx]), *receivers[r] );
                }

                // Do the test with varying numbers of senders
                harness_counting_sender<InputType> *senders = NULL;
                for (size_t num_senders = 1; num_senders <= MAX_NODES; ++num_senders ) {
                    // Create num_senders senders, set their message limit each to N, and connect them to the exe_vec[node_idx]
                    senders = new harness_counting_sender<InputType>[num_senders];
                    for (size_t s = 0; s < num_senders; ++s ) {
                        senders[s].my_limit = N;
                        tbb::flow::make_edge( senders[s], exe_vec[node_idx] );
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
                        ASSERT( senders[s].my_receiver == &exe_vec[node_idx], NULL );
                    }
                    // validate the receivers
                    for (size_t r = 0; r < num_receivers; ++r ) {
                        receivers[r]->validate();
                    }
                    delete [] senders;
                }
                for (size_t r = 0; r < num_receivers; ++r ) {
                    tbb::flow::remove_edge( tbb::flow::output_port<0>(exe_vec[node_idx]), *receivers[r] );
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
            }
        }
    }
}

const size_t Offset = 123;
tbb::atomic<size_t> global_execute_count;

struct inc_functor {

    tbb::atomic<size_t> local_execute_count;
    inc_functor( ) { local_execute_count = 0; }
    inc_functor( const inc_functor &f ) { local_execute_count = f.local_execute_count; }

    template<typename output_ports_type>
    void operator()( int i, output_ports_type &p ) {
       ++global_execute_count;
       ++local_execute_count;
       (void)tbb::flow::get<0>(p).try_put(i);
    }

};

template< typename InputType, typename OutputTuple >
void buffered_levels_with_copy( size_t concurrency ) {
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type OutputType;
    // Do for lc = 1 to concurrency level
    for ( size_t lc = 1; lc <= concurrency; ++lc ) {
        tbb::flow::graph g;

        inc_functor cf;
        cf.local_execute_count = Offset;
        global_execute_count = Offset;

        tbb::flow::multifunction_node< InputType, OutputTuple > exe_node( g, lc, cf );

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
           
            std::vector< harness_mapped_receiver<OutputType>* > receivers(num_receivers);
            for (size_t i = 0; i < num_receivers; i++) {
                receivers[i] = new harness_mapped_receiver<OutputType>(g);
            }

            for (size_t r = 0; r < num_receivers; ++r ) {
               tbb::flow::make_edge( tbb::flow::output_port<0>(exe_node), *receivers[r] );
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
                tbb::flow::remove_edge( tbb::flow::output_port<0>(exe_node), *receivers[r] );
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
    }
}

template< typename InputType, typename OutputTuple >
void run_buffered_levels( int c ) {
    #if __TBB_CPP11_LAMBDAS_PRESENT
    typedef typename tbb::flow::multifunction_node<InputType,OutputTuple>::output_ports_type output_ports_type;
    buffered_levels<InputType,OutputTuple>( c, []( InputType i, output_ports_type &p ) { harness_graph_multifunction_executor<InputType, OutputTuple>::func(i,p); } );
    #endif
    buffered_levels<InputType,OutputTuple>( c, &harness_graph_multifunction_executor<InputType, OutputTuple>::func );
    buffered_levels<InputType,OutputTuple>( c, typename harness_graph_multifunction_executor<InputType, OutputTuple>::functor() );
    buffered_levels_with_copy<InputType,OutputTuple>( c );
}


//! Performs test on executable nodes with limited concurrency
/** Theses tests check:
    1) that the nodes will accepts puts up to the concurrency limit,
    2) the nodes do not exceed the concurrency limit even when run with more threads (this is checked in the harness_graph_executor),
    3) the nodes will receive puts from multiple successors simultaneously,
    and 4) the nodes will send to multiple predecessors.
    There is no checking of the contents of the messages for corruption.
*/

template< typename InputType, typename OutputTuple, typename Body >
void concurrency_levels( size_t concurrency, Body body ) {
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type OutputType;
    for ( size_t lc = 1; lc <= concurrency; ++lc ) {
        tbb::flow::graph g;

        // Set the execute_counter back to zero in the harness
        harness_graph_multifunction_executor<InputType, OutputTuple>::execute_count = 0;
        // Set the number of current executors to zero.
        harness_graph_multifunction_executor<InputType, OutputTuple>::current_executors = 0;
        // Set the max allowed executors to lc.  There is a check in the functor to make sure this is never exceeded.
        harness_graph_multifunction_executor<InputType, OutputTuple>::max_executors = lc;


        tbb::flow::multifunction_node< InputType, OutputTuple, tbb::flow::rejecting > exe_node( g, lc, body );

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {

            std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));

            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::make_edge( tbb::flow::output_port<0>(exe_node), receivers[r] );
            }

            harness_counting_sender<InputType> *senders = NULL;

            for (size_t num_senders = 1; num_senders <= MAX_NODES; ++num_senders ) {
                {
                    // Exclusively lock m to prevent exe_node from finishing
                    tbb::spin_rw_mutex::scoped_lock l( harness_graph_multifunction_executor< InputType, OutputTuple>::template mutex_holder<tbb::spin_rw_mutex>::mutex );

                    // put to lc level, it will accept and then block at m
                    for ( size_t c = 0 ; c < lc ; ++c ) {
                        ASSERT( exe_node.try_put( InputType() ) == true, NULL );
                    }
                    // it only accepts to lc level
                    ASSERT( exe_node.try_put( InputType() ) == false, NULL );

                    senders = new harness_counting_sender<InputType>[num_senders];
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
                tbb::flow::remove_edge( tbb::flow::output_port<0>(exe_node), receivers[r] );
            }
            ASSERT( exe_node.try_put( InputType() ) == true, NULL );
            g.wait_for_all();
            for (size_t r = 0; r < num_receivers; ++r ) {
                ASSERT( int(receivers[r].my_count) == 0, NULL );
            }
        }
    }
}

template< typename InputType, typename OutputTuple >
void run_concurrency_levels( int c ) {
    #if __TBB_CPP11_LAMBDAS_PRESENT
    typedef typename tbb::flow::multifunction_node<InputType,OutputTuple>::output_ports_type output_ports_type;
    concurrency_levels<InputType,OutputTuple>( c, []( InputType i, output_ports_type &p ) { harness_graph_multifunction_executor<InputType, OutputTuple>::template tfunc<tbb::spin_rw_mutex>(i,p); } );
    #endif
    concurrency_levels<InputType,OutputTuple>( c, &harness_graph_multifunction_executor<InputType, OutputTuple>::template tfunc<tbb::spin_rw_mutex> );
    concurrency_levels<InputType,OutputTuple>( c, typename harness_graph_multifunction_executor<InputType, OutputTuple>::template tfunctor<tbb::spin_rw_mutex>() );
}


struct empty_no_assign {
   empty_no_assign() {}
   empty_no_assign( int ) {}
   operator int() { return 0; }
   operator int() const { return 0; }
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
/** These tests check:
    1) that the nodes will accept all puts
    2) the nodes will receive puts from multiple predecessors simultaneously,
    and 3) the nodes will send to multiple successors.
    There is no checking of the contents of the messages for corruption.
*/

template< typename InputType, typename OutputTuple, typename Body >
void unlimited_concurrency( Body body ) {
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type OutputType;

    for (int p = 1; p < 2*MaxThread; ++p) {
        tbb::flow::graph g;
        tbb::flow::multifunction_node< InputType, OutputTuple, tbb::flow::rejecting > exe_node( g, tbb::flow::unlimited, body );

        for (size_t num_receivers = 1; num_receivers <= MAX_NODES; ++num_receivers ) {
            std::vector< harness_counting_receiver<OutputType> > receivers(num_receivers, harness_counting_receiver<OutputType>(g));

            harness_graph_multifunction_executor<InputType, OutputTuple>::execute_count = 0;

            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::make_edge( tbb::flow::output_port<0>(exe_node), receivers[r] );
            }

            NativeParallelFor( p, parallel_puts<InputType>(exe_node) );
            g.wait_for_all();

            // 2) the nodes will receive puts from multiple predecessors simultaneously,
            size_t ec = harness_graph_multifunction_executor<InputType, OutputTuple>::execute_count;
            ASSERT( (int)ec == p*N, NULL );
            for (size_t r = 0; r < num_receivers; ++r ) {
                size_t c = receivers[r].my_count;
                // 3) the nodes will send to multiple successors.
                ASSERT( (int)c == p*N, NULL );
            }
            for (size_t r = 0; r < num_receivers; ++r ) {
                tbb::flow::remove_edge( tbb::flow::output_port<0>(exe_node), receivers[r] );
            }
        }
    }
}

template< typename InputType, typename OutputTuple >
void run_unlimited_concurrency() {
    harness_graph_multifunction_executor<InputType, OutputTuple>::max_executors = 0;
    #if __TBB_CPP11_LAMBDAS_PRESENT
    typedef typename tbb::flow::multifunction_node<InputType,OutputTuple>::output_ports_type output_ports_type;
    unlimited_concurrency<InputType,OutputTuple>( []( InputType i, output_ports_type &p ) { harness_graph_multifunction_executor<InputType, OutputTuple>::func(i,p); } );
    #endif
    unlimited_concurrency<InputType,OutputTuple>( &harness_graph_multifunction_executor<InputType, OutputTuple>::func );
    unlimited_concurrency<InputType,OutputTuple>( typename harness_graph_multifunction_executor<InputType, OutputTuple>::functor() );
}

template<typename InputType, typename OutputTuple>
struct oddEvenBody {
    typedef typename tbb::flow::multifunction_node<InputType,OutputTuple>::output_ports_type output_ports_type;
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type EvenType;
    typedef typename tbb::flow::tuple_element<1,OutputTuple>::type OddType;
    void operator() (const InputType &i, output_ports_type &p) {
        if((int)i % 2) {
            (void)tbb::flow::get<1>(p).try_put(OddType(i));
        }
        else {
            (void)tbb::flow::get<0>(p).try_put(EvenType(i));
        }
    }
};

template<typename InputType, typename OutputTuple >
void run_multiport_test(int num_threads) {
    typedef typename tbb::flow::multifunction_node<InputType, OutputTuple> mo_node_type;
    typedef typename tbb::flow::tuple_element<0,OutputTuple>::type EvenType;
    typedef typename tbb::flow::tuple_element<1,OutputTuple>::type OddType;
    tbb::task_scheduler_init init(num_threads);
    tbb::flow::graph g;
    mo_node_type mo_node(g, tbb::flow::unlimited, oddEvenBody<InputType, OutputTuple>() );

    tbb::flow::queue_node<EvenType> q0(g);
    tbb::flow::queue_node<OddType> q1(g);

    tbb::flow::make_edge(tbb::flow::output_port<0>(mo_node), q0);
    tbb::flow::make_edge(tbb::flow::output_port<1>(mo_node), q1);

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    ASSERT(mo_node.predecessor_count() == 0, NULL);
    ASSERT(tbb::flow::output_port<0>(mo_node).successor_count() == 1, NULL);
    typedef typename mo_node_type::output_ports_type oports_type;
    typedef typename tbb::flow::tuple_element<0,oports_type>::type port0_type;
    typename port0_type::successor_list_type my_0succs;
    tbb::flow::output_port<0>(mo_node).copy_successors(my_0succs);
    ASSERT(my_0succs.size() == 1, NULL);
    typename mo_node_type::predecessor_list_type my_preds;
    mo_node.copy_predecessors(my_preds);
    ASSERT(my_preds.size() == 0, NULL);
#endif

    for(InputType i = 0; i < N; ++i) {
        mo_node.try_put(i);
    }

    g.wait_for_all();
    for(int i = 0; i < N/2; ++i) {
        EvenType e;
        OddType o;
        ASSERT(q0.try_get(e) && (int)e % 2 == 0, NULL);
        ASSERT(q1.try_get(o) && (int)o % 2 == 1, NULL);
    }
}

//! Tests limited concurrency cases for nodes that accept data messages
void test_concurrency(int num_threads) {
    tbb::task_scheduler_init init(num_threads);
    run_concurrency_levels<int,tbb::flow::tuple<int> >(num_threads);
    run_concurrency_levels<int,tbb::flow::tuple<tbb::flow::continue_msg> >(num_threads);
    run_buffered_levels<int, tbb::flow::tuple<int> >(num_threads);
    run_unlimited_concurrency<int, tbb::flow::tuple<int> >();
    run_unlimited_concurrency<int,tbb::flow::tuple<empty_no_assign> >();
    run_unlimited_concurrency<empty_no_assign,tbb::flow::tuple<int> >();
    run_unlimited_concurrency<empty_no_assign,tbb::flow::tuple<empty_no_assign> >();
    run_unlimited_concurrency<int,tbb::flow::tuple<tbb::flow::continue_msg> >();
    run_unlimited_concurrency<empty_no_assign,tbb::flow::tuple<tbb::flow::continue_msg> >();
    run_multiport_test<int, tbb::flow::tuple<int, int> >(num_threads);
    run_multiport_test<float, tbb::flow::tuple<int, double> >(num_threads);
}

template<typename Policy>
void test_ports_return_references() {
    tbb::flow::graph g;
    typedef int InputType;
    typedef tbb::flow::tuple<int> OutputTuple;
    tbb::flow::multifunction_node<InputType, OutputTuple, Policy> mf_node(
        g, tbb::flow::unlimited,
        &harness_graph_multifunction_executor<InputType, OutputTuple>::empty_func );
    test_output_ports_return_ref(mf_node);
}

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
// the integer received indicates which output ports should succeed and which should fail
// on try_put().
typedef tbb::flow::multifunction_node<int, tbb::flow::tuple<int, int> > mf_node;

struct add_to_counter {
    int my_invocations;
    int *counter;
    add_to_counter(int& var):counter(&var){ my_invocations = 0;}
    void operator()(const int &i, mf_node::output_ports_type &outports) {
        *counter+=1;
        ++my_invocations;
        if(i & 0x1) {
            ASSERT(tbb::flow::get<0>(outports).try_put(i), "port 0 expected to succeed");
        }
        else {
            ASSERT(!tbb::flow::get<0>(outports).try_put(i), "port 0 expected to fail");
        }
        if(i & 0x2) {
            ASSERT(tbb::flow::get<1>(outports).try_put(i), "port 1 expected to succeed");
        }
        else {
            ASSERT(!tbb::flow::get<1>(outports).try_put(i), "port 1 expected to fail");
        }
    }
    int my_inner() { return my_invocations; }
};

template<class FTYPE>
void test_extract() {
    int my_count = 0;
    int cm;
    tbb::flow::graph g;
    tbb::flow::broadcast_node<int> b0(g);
    tbb::flow::broadcast_node<int> b1(g);
    tbb::flow::multifunction_node<int, tbb::flow::tuple<int,int>, FTYPE> mf0(g, tbb::flow::unlimited, add_to_counter(my_count));
    tbb::flow::queue_node<int> q0(g);
    tbb::flow::queue_node<int> q1(g);

    tbb::flow::make_edge(b0, mf0);
    tbb::flow::make_edge(b1, mf0);
    tbb::flow::make_edge(tbb::flow::output_port<0>(mf0), q0);
    tbb::flow::make_edge(tbb::flow::output_port<1>(mf0), q1);
    for( int i = 0; i < 2; ++i ) {

        /* b0          */
        /*   \   |--q0 */
        /*    mf0+     */
        /*   /   |--q1 */
        /* b1          */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(mf0.predecessor_count() == 2
                && tbb::flow::output_port<0>(mf0).successor_count() == 1
                && tbb::flow::output_port<1>(mf0).successor_count() == 1
                , "mf0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");
        ASSERT(q1.predecessor_count() == 1 && q1.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(3);
        g.wait_for_all();
        ASSERT(my_count == 1, "multifunction_node didn't fire");
        ASSERT(q0.try_get(cm), "multifunction_node didn't forward to 0");
        ASSERT(q1.try_get(cm), "multifunction_node didn't forward to 1");
        b1.try_put(3);
        g.wait_for_all();
        ASSERT(my_count == 2, "multifunction_node didn't fire");
        ASSERT(q0.try_get(cm), "multifunction_node didn't forward to 0");
        ASSERT(q1.try_get(cm), "multifunction_node didn't forward to 1");

        b0.extract();


        /* b0          */
        /*       |--q0 */
        /*    mf0+     */
        /*   /   |--q1 */
        /* b1          */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(mf0.predecessor_count() == 1
                && tbb::flow::output_port<0>(mf0).successor_count() == 1
                && tbb::flow::output_port<1>(mf0).successor_count() == 1
                , "mf0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 1 && q0.successor_count() == 0, "q0 has incorrect counts");
        ASSERT(q1.predecessor_count() == 1 && q1.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(1);
        b0.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 2, "b0 messages being forwarded to multifunction_node even though it is disconnected");
        b1.try_put(3);
        g.wait_for_all();
        ASSERT(my_count == 3, "multifunction_node didn't fire though it has only one predecessor");
        ASSERT(q0.try_get(cm), "multifunction_node didn't forward second time");
        ASSERT(q1.try_get(cm), "multifunction_node didn't forward second time");

        q0.extract();

        /* b0          */
        /*       |  q0 */
        /*    mf0+     */
        /*   /   |--q1 */
        /* b1          */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 1, "b1 has incorrect counts");
        ASSERT(mf0.predecessor_count() == 1
                && tbb::flow::output_port<0>(mf0).successor_count() == 0
                && tbb::flow::output_port<1>(mf0).successor_count() == 1
                , "mf0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");
        ASSERT(q1.predecessor_count() == 1 && q1.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(1);
        b0.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 3, "b0 messages being forwarded to multifunction_node even though it is disconnected");
        b1.try_put(2);
        g.wait_for_all();
        ASSERT(my_count == 4, "multifunction_node didn't fire though it has one predecessor");
        ASSERT(!q0.try_get(cm), "multifunction_node forwarded");
        ASSERT(q1.try_get(cm), "multifunction_node forwarded");
        mf0.extract();

        if(i == 0) {
        }
        else {
            g.reset(tbb::flow::rf_reset_bodies);
        }


        /* b0          */
        /*       |  q0 */
        /*    mf0+     */
        /*       |  q1 */
        /* b1          */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 0, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(mf0.predecessor_count() == 0
                && tbb::flow::output_port<0>(mf0).successor_count() == 0
                && tbb::flow::output_port<1>(mf0).successor_count() == 0
                , "mf0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");
        ASSERT(q1.predecessor_count() == 0 && q1.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(1);
        b0.try_put(1);
        g.wait_for_all();
        ASSERT(my_count == 4, "b0 messages being forwarded to multifunction_node even though it is disconnected");
        b1.try_put(2);
        g.wait_for_all();
        ASSERT(my_count == 4, "b1 messages being forwarded to multifunction_node even though it is disconnected");
        ASSERT(!q0.try_get(cm), "multifunction_node forwarded");
        ASSERT(!q1.try_get(cm), "multifunction_node forwarded");
        make_edge(b0, mf0);

        /* b0          */
        /*   \   |  q0 */
        /*    mf0+     */
        /*       |  q1 */
        /* b1          */

        ASSERT(b0.predecessor_count() == 0 && b0.successor_count() == 1, "b0 has incorrect counts");
        ASSERT(b1.predecessor_count() == 0 && b1.successor_count() == 0, "b1 has incorrect counts");
        ASSERT(mf0.predecessor_count() == 1
                && tbb::flow::output_port<0>(mf0).successor_count() == 0
                && tbb::flow::output_port<1>(mf0).successor_count() == 0
                , "mf0 has incorrect counts");
        ASSERT(q0.predecessor_count() == 0 && q0.successor_count() == 0, "q0 has incorrect counts");
        ASSERT(q1.predecessor_count() == 0 && q1.successor_count() == 0, "q0 has incorrect counts");
        b0.try_put(0);
        g.wait_for_all();
        ASSERT(my_count == 5, "multifunction_node didn't fire though it has one predecessor");
        b1.try_put(2);
        g.wait_for_all();
        ASSERT(my_count == 5, "multifunction_node fired though it has only one predecessor");
        ASSERT(!q0.try_get(cm), "multifunction_node forwarded");
        ASSERT(!q1.try_get(cm), "multifunction_node forwarded");

        tbb::flow::make_edge(b1, mf0);
        tbb::flow::make_edge(tbb::flow::output_port<0>(mf0), q0);
        tbb::flow::make_edge(tbb::flow::output_port<1>(mf0), q1);
        ASSERT( ( i == 0 && tbb::flow::copy_body<add_to_counter>(mf0).my_inner() == 5 ) ||
               ( i == 1 && tbb::flow::copy_body<add_to_counter>(mf0).my_inner() == 1 ) , "reset_bodies failed");
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
    test_ports_return_references<tbb::flow::queueing>();
    test_ports_return_references<tbb::flow::rejecting>();

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    test_extract<tbb::flow::rejecting>();
    test_extract<tbb::flow::queueing>();
#endif
   return Harness::Done;
}
