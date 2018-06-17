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
#include "harness_barrier.h"
#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"

const int T = 4;
const int W = 4;

struct decrement_wait : NoAssign {

    tbb::flow::graph * const my_graph;
    bool * const my_done_flag;

    decrement_wait( tbb::flow::graph &h, bool *done_flag ) : my_graph(&h), my_done_flag(done_flag) {}

    void operator()(int i) const {
        Harness::Sleep(10*i);
        my_done_flag[i] = true;
        my_graph->decrement_wait_count();
    }
};

static void test_wait_count() {
   tbb::flow::graph h;
   for (int i = 0; i < T; ++i ) {
       bool done_flag[W];
       for (int j = 0; j < W; ++j ) {
           for ( int w = 0; w < W; ++w ) done_flag[w] = false;
           for ( int w = 0; w < j; ++w ) h.increment_wait_count();

           NativeParallelFor( j, decrement_wait(h, done_flag) );
           h.wait_for_all();
           for ( int w = 0; w < W; ++w ) {
              if ( w < j ) ASSERT( done_flag[w] == true, NULL );
              else ASSERT( done_flag[w] == false, NULL );
           }
       }
   }
}

const int F = 100;

#if __TBB_CPP11_LAMBDAS_PRESENT
bool lambda_flag[F];
#endif
bool functor_flag[F];

struct set_functor {
    int my_i;
    set_functor( int i ) : my_i(i) {}
    void operator()() { functor_flag[my_i] = true; }
};

struct return_functor {
    int my_i;
    return_functor( int i ) : my_i(i) {}
    int operator()() { return my_i; }
};

static void test_run() {
    tbb::flow::graph h;
    for (int i = 0; i < T; ++i ) {

        // Create receivers and flag arrays
        #if __TBB_CPP11_LAMBDAS_PRESENT
        harness_mapped_receiver<int> lambda_r(h);
        lambda_r.initialize_map( F, 1 );
        #endif
        harness_mapped_receiver<int> functor_r(h);
        functor_r.initialize_map( F, 1 );

        // Initialize flag arrays
        for (int j = 0; j < F; ++j ) {
            #if __TBB_CPP11_LAMBDAS_PRESENT
            lambda_flag[j] = false;
            #endif
            functor_flag[j] = false;
        }

        for ( int j = 0; j < F; ++j ) {
            #if __TBB_CPP11_LAMBDAS_PRESENT
                h.run( [=]() { lambda_flag[j] = true; } );
                h.run( lambda_r, [=]() { return j; } );
            #endif
            h.run( set_functor(j) );
            h.run( functor_r, return_functor(j) );
        }
        h.wait_for_all();
        for ( int j = 0; j < F; ++j ) {
        #if __TBB_CPP11_LAMBDAS_PRESENT
            ASSERT( lambda_flag[i] == true, NULL );
        #endif
            ASSERT( functor_flag[i] == true, NULL );
        }
        #if __TBB_CPP11_LAMBDAS_PRESENT
        lambda_r.validate();
        #endif
        functor_r.validate();
    }
}

// Encapsulate object we want to store in vector (because contained type must have
// copy constructor and assignment operator
class my_int_buffer {
    tbb::flow::buffer_node<int> *b;
    tbb::flow::graph& my_graph;
public:
    my_int_buffer(tbb::flow::graph &g) : my_graph(g) { b = new tbb::flow::buffer_node<int>(my_graph); }
    my_int_buffer(const my_int_buffer& other) : my_graph(other.my_graph) {
        b = new tbb::flow::buffer_node<int>(my_graph);
    }
    ~my_int_buffer() { delete b; }
    my_int_buffer& operator=(const my_int_buffer& /*other*/) {
        return *this;
    }
};

// test the graph iterator, delete nodes from graph, test again
void test_iterator() {
   tbb::flow::graph g;
   my_int_buffer a_buffer(g);
   my_int_buffer b_buffer(g);
   my_int_buffer c_buffer(g);
   my_int_buffer *d_buffer = new my_int_buffer(g);
   my_int_buffer e_buffer(g);
   std::vector< my_int_buffer > my_buffer_vector(10, c_buffer);

   int count = 0;
   for (tbb::flow::graph::iterator it = g.begin(); it != g.end(); ++it) {
       count++;
   }
   ASSERT(count==15, "error in iterator count");

   delete d_buffer;

   count = 0;
   for (tbb::flow::graph::iterator it = g.begin(); it != g.end(); ++it) {
       count++;
   }
   ASSERT(count==14, "error in iterator count");

   my_buffer_vector.clear();

   count = 0;
   for (tbb::flow::graph::iterator it = g.begin(); it != g.end(); ++it) {
       count++;
   }
   ASSERT(count==4, "error in iterator count");
}

class AddRemoveBody : NoAssign {
    tbb::flow::graph& g;
    int nThreads;
    Harness::SpinBarrier &barrier;
public:
    AddRemoveBody(int nthr, Harness::SpinBarrier &barrier_, tbb::flow::graph& _g) :
        g(_g), nThreads(nthr), barrier(barrier_)
    {}
    void operator()(const int /*threadID*/) const {
        my_int_buffer b(g);
        {
            std::vector<my_int_buffer> my_buffer_vector(100, b);
            barrier.wait();  // wait until all nodes are created
            // now test that the proper number of nodes were created
            int count = 0;
            for (tbb::flow::graph::iterator it = g.begin(); it != g.end(); ++it) {
                count++;
            }
            ASSERT(count==101*nThreads, "error in iterator count");
            barrier.wait();  // wait until all threads are done counting
        } // all nodes but for the initial node on this thread are deleted
        barrier.wait(); // wait until all threads have deleted all nodes in their vectors
        // now test that all the nodes were deleted except for the initial node
        int count = 0;
        for (tbb::flow::graph::iterator it = g.begin(); it != g.end(); ++it) {
            count++;
        }
        ASSERT(count==nThreads, "error in iterator count");
        barrier.wait();  // wait until all threads are done counting
    } // initial node gets deleted
};

void test_parallel(int nThreads) {
    tbb::flow::graph g;
    Harness::SpinBarrier barrier(nThreads);
    AddRemoveBody body(nThreads, barrier, g);
    NativeParallelFor(nThreads, body);
}

/*
 * Functors for graph arena spawn tests
 */

inline void check_arena(tbb::task_arena* a) {
    ASSERT(a->max_concurrency() == 2, NULL);
    ASSERT(tbb::this_task_arena::max_concurrency() == 1, NULL);
}

struct run_functor {
    tbb::task_arena* my_a;
    int return_value;
    run_functor(tbb::task_arena* a) : my_a(a), return_value(1) {}
    int operator()() {
        check_arena(my_a);
        return return_value;
    }
};

template < typename T >
struct function_body {
    tbb::task_arena* my_a;
    function_body(tbb::task_arena* a) : my_a(a) {}
    tbb::flow::continue_msg operator()(const T& /*arg*/) {
        check_arena(my_a);
        return tbb::flow::continue_msg();
    }
};

typedef tbb::flow::multifunction_node< int, tbb::flow::tuple< int > > mf_node;

struct multifunction_body {
    tbb::task_arena* my_a;
    multifunction_body(tbb::task_arena* a) : my_a(a) {}
    void operator()(const int& /*arg*/, mf_node::output_ports_type& /*outports*/) {
        check_arena(my_a);
    }
};

struct source_body {
    tbb::task_arena* my_a;
    int counter;
    source_body(tbb::task_arena* a) : my_a(a), counter(0) {}
    bool operator()(const int& /*i*/) {
        check_arena(my_a);
        if (counter < 1) {
          ++counter;
          return true;
       }
       return false;
    }
};

struct run_test_functor : tbb::internal::no_assign {
    tbb::task_arena* fg_arena;
    tbb::flow::graph& my_graph;

    run_test_functor(tbb::task_arena* a, tbb::flow::graph& g) : fg_arena(a), my_graph(g) {}
    void operator()() const {
        harness_mapped_receiver<int> functor_r(my_graph);
        functor_r.initialize_map(F, 1);

        my_graph.run(run_functor(fg_arena));
        my_graph.run(functor_r, run_functor(fg_arena));

        my_graph.wait_for_all();
    }
};

struct nodes_test_functor : tbb::internal::no_assign {
    tbb::task_arena* fg_arena;
    tbb::flow::graph& my_graph;

    nodes_test_functor(tbb::task_arena* a, tbb::flow::graph& g) : fg_arena(a), my_graph(g) {}
    void operator()() const {

        // Define test nodes
        // Continue, function, source nodes
        tbb::flow::continue_node< tbb::flow::continue_msg > c_n(my_graph, function_body<tbb::flow::continue_msg>(fg_arena));
        tbb::flow::function_node< int > f_n(my_graph, tbb::flow::unlimited, function_body<int>(fg_arena));
        tbb::flow::source_node< int > s_n(my_graph, source_body(fg_arena), false);

        // Multifunction node
        mf_node m_n(my_graph, tbb::flow::unlimited, multifunction_body(fg_arena));

        // Join node
        tbb::flow::function_node< tbb::flow::tuple< int, int > > join_f_n(my_graph, tbb::flow::unlimited, function_body< tbb::flow::tuple< int, int > >(fg_arena));
        tbb::flow::join_node< tbb::flow::tuple< int, int > > j_n(my_graph);
        make_edge(j_n, join_f_n);

        // Split node
        tbb::flow::function_node< int > split_f_n1 = f_n;
        tbb::flow::function_node< int > split_f_n2 = f_n;
        tbb::flow::split_node< tbb::flow::tuple< int, int > > sp_n(my_graph);
        make_edge(tbb::flow::output_port<0>(sp_n), split_f_n1);
        make_edge(tbb::flow::output_port<1>(sp_n), split_f_n2);

        // Overwrite node
        tbb::flow::function_node< int > ow_f_n = f_n;
        tbb::flow::overwrite_node< int > ow_n(my_graph);
        make_edge(ow_n, ow_f_n);

        // Write once node
        tbb::flow::function_node< int > w_f_n = f_n;
        tbb::flow::write_once_node< int > w_n(my_graph);
        make_edge(w_n, w_f_n);

        // Buffer node
        tbb::flow::function_node< int > buf_f_n = f_n;
        tbb::flow::buffer_node< int > buf_n(my_graph);
        make_edge(w_n, buf_f_n);

        // Limiter node
        tbb::flow::function_node< int > l_f_n = f_n;
        tbb::flow::limiter_node< int > l_n(my_graph, 1);
        make_edge(l_n, l_f_n);

        // Execute nodes
        c_n.try_put( tbb::flow::continue_msg() );
        f_n.try_put(1);
        m_n.try_put(1);
        s_n.activate();

        tbb::flow::input_port<0>(j_n).try_put(1);
        tbb::flow::input_port<1>(j_n).try_put(1);

        tbb::flow::tuple< int, int > sp_tuple(1, 1);
        sp_n.try_put(sp_tuple);

        ow_n.try_put(1);
        w_n.try_put(1);
        buf_n.try_put(1);
        l_n.try_put(1);

        my_graph.wait_for_all();
    }
};

void test_graph_arena() {
    // There is only one thread for execution (master thread).
    // So, if graph's tasks get spawned in different arena
    // master thread won't be able to find them in its own arena.
    // In this case test should hang.
    tbb::task_scheduler_init init(1);

    tbb::flow::graph g;
    tbb::task_arena fg_arena;
    fg_arena.initialize(2);
    fg_arena.execute(run_test_functor(&fg_arena, g));
    fg_arena.execute(nodes_test_functor(&fg_arena, g));
}

int TestMain() {
    if( MinThread<1 ) {
        REPORT("number of threads must be positive\n");
        exit(1);
    }
    for( int p=MinThread; p<=MaxThread; ++p ) {
       tbb::task_scheduler_init init(p);
       test_wait_count();
       test_run();
       test_iterator();
       test_parallel(p);
   }
   test_graph_arena();
   return Harness::Done;
}
