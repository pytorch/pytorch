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

#include "harness.h"
#if __TBB_FLOW_GRAPH_CPP11_FEATURES

#include "tbb/flow_graph.h"
#include "harness_graph.h"
#include <tuple>
#include <cmath>
#include <vector>

struct passthru_body {
    int operator()( int i ) {
        return i;
    }
};

class src_body{
    int start;
    int finish;
    int step;
public:
    src_body(int f, int s) : start(1), finish(f), step(s) {}
    bool operator()(int &a) {
       a = start;
       if (start <= finish) {
           a = start;
           start+=step;
           return true;
       }
       else {
           return false;
       };
   }
};

struct m_fxn_body{
    void operator()(int, tbb::flow::multifunction_node<int, tbb::flow::tuple<int,int> >::output_ports_type ) {}
};

struct ct_body {
ct_body(){}
    void operator()(tbb::flow::continue_msg){}
};

struct seq_body {
int operator()(int i){return i;}
};

template<int N, typename T1, typename T2>
struct compare {
    static void compare_refs(T1 tuple1, T2 tuple2) {
    ASSERT( &tbb::flow::get<N>(tuple1) == &tbb::flow::get<N>(tuple2), "ports not set correctly");
    compare<N-1, T1, T2>::compare_refs(tuple1, tuple2);
    }
};

template<typename T1, typename T2>
struct compare<1, T1, T2> {
    static void compare_refs(T1 tuple1, T2 tuple2) {
    ASSERT(&tbb::flow::get<0>(tuple1) == &tbb::flow::get<0>(tuple2), "port 0 not correctly set");
    }
};

void add_all_nodes (){
    tbb::flow::graph g;

    typedef tbb::flow::tuple<tbb::flow::continue_msg, tbb::flow::tuple<int, int>, int, int, int, int,
                             int, int, int, int, int, int, int, int > InputTupleType;

    typedef tbb::flow::tuple<tbb::flow::continue_msg, tbb::flow::tuple<int, int>, tbb::flow::tagged_msg<size_t, int, float>,
                             int, int, int, int, int, int, int, int, int, int, int, int >  OutputTupleType;

    typedef tbb::flow::tuple< > EmptyTupleType;

    typedef tbb::flow::composite_node<InputTupleType, OutputTupleType > input_output_type;
    typedef tbb::flow::composite_node<InputTupleType, EmptyTupleType > input_only_type;
    typedef tbb::flow::composite_node<EmptyTupleType, OutputTupleType > output_only_type;

    const size_t NUM_INPUTS = tbb::flow::tuple_size<InputTupleType>::value;
    const size_t NUM_OUTPUTS = tbb::flow::tuple_size<OutputTupleType>::value;

    //node types
    tbb::flow::continue_node<tbb::flow::continue_msg> ct(g, ct_body());
    tbb::flow::split_node< tbb::flow::tuple<int, int> > s(g);
    tbb::flow::source_node<int> src(g, src_body(20,5), false);
    tbb::flow::function_node<int, int> fxn(g, tbb::flow::unlimited, passthru_body());
    tbb::flow::multifunction_node<int, tbb::flow::tuple<int, int> > m_fxn(g, tbb::flow::unlimited, m_fxn_body());
    tbb::flow::broadcast_node<int> bc(g);
    tbb::flow::limiter_node<int> lim(g, 2);
    tbb::flow::indexer_node<int, float> ind(g);
    tbb::flow::join_node< tbb::flow::tuple< int, int >, tbb::flow::queueing > j(g);
    tbb::flow::queue_node<int> q(g);
    tbb::flow::buffer_node<int> bf(g);
    tbb::flow::priority_queue_node<int> pq(g);
    tbb::flow::write_once_node<int> wo(g);
    tbb::flow::overwrite_node<int> ovw(g);
    tbb::flow::sequencer_node<int> seq(g, seq_body());

#if !__TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN
    auto input_tuple = std::tie(ct, s, m_fxn, fxn, bc, tbb::flow::input_port<0>(j), lim, q, tbb::flow::input_port<0>(ind),
                                pq, ovw, wo, bf, seq);
    auto output_tuple = std::tie(ct,j, ind, fxn, src, bc, tbb::flow::output_port<0>(s), lim, tbb::flow::output_port<0>(m_fxn),
                                 q, pq, ovw, wo, bf, seq );
#else
    // upcasting from derived to base for a tuple of references created by std::tie
    // fails on gcc 4.4 (and all icc in that environment)
    input_output_type::input_ports_type input_tuple(ct, s, m_fxn, fxn, bc, tbb::flow::input_port<0>(j), lim, q,
                                                    tbb::flow::input_port<0>(ind), pq, ovw, wo, bf, seq);

    input_output_type::output_ports_type output_tuple(ct,j, ind, fxn, src, bc, tbb::flow::output_port<0>(s),
                                                      lim, tbb::flow::output_port<0>(m_fxn), q, pq, ovw, wo, bf, seq);
#endif

    //composite_node with both input_ports and output_ports
    input_output_type a_node(g);
    a_node.set_external_ports(input_tuple, output_tuple);

    a_node.add_visible_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);
    a_node.add_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);

    auto a_node_input_ports_ptr = a_node.input_ports();
    compare<NUM_INPUTS-1, decltype(a_node_input_ports_ptr), decltype(input_tuple)>::compare_refs(a_node_input_ports_ptr, input_tuple);
    ASSERT (NUM_INPUTS == tbb::flow::tuple_size<decltype(a_node_input_ports_ptr)>::value, "not all declared input ports were bound to nodes");

    auto a_node_output_ports_ptr = a_node.output_ports();
    compare<NUM_OUTPUTS-1, decltype(a_node_output_ports_ptr), decltype(output_tuple)>::compare_refs(a_node_output_ports_ptr, output_tuple);
    ASSERT(NUM_OUTPUTS == tbb::flow::tuple_size<decltype(a_node_output_ports_ptr)>::value, "not all declared output ports were bound to nodes");

    //composite_node with only input_ports
    input_only_type b_node(g);
    b_node.set_external_ports(input_tuple);

    b_node.add_visible_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);
    b_node.add_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);

    auto b_node_input_ports_ptr = b_node.input_ports();
    compare<NUM_INPUTS-1, decltype(b_node_input_ports_ptr), decltype(input_tuple)>::compare_refs(b_node_input_ports_ptr, input_tuple);
    ASSERT (NUM_INPUTS == tbb::flow::tuple_size<decltype(b_node_input_ports_ptr)>::value, "not all declared input ports were bound to nodes");

    //composite_node with only output_ports
    output_only_type c_node(g);
    c_node.set_external_ports(output_tuple);

    c_node.add_visible_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);

    c_node.add_nodes(src, fxn, m_fxn, bc, lim, ind, s, ct, j, q, bf, pq, wo, ovw, seq);

    auto c_node_output_ports_ptr = c_node.output_ports();
    compare<NUM_OUTPUTS-1, decltype(c_node_output_ports_ptr), decltype(output_tuple)>::compare_refs(c_node_output_ports_ptr, output_tuple);
    ASSERT (NUM_OUTPUTS == tbb::flow::tuple_size<decltype(c_node_output_ports_ptr)>::value, "not all declared input ports were bound to nodes");
}

struct tiny_node : public tbb::flow::composite_node< tbb::flow::tuple< int >, tbb::flow::tuple< int > > {
    tbb::flow::function_node< int, int > f1;
    tbb::flow::function_node< int, int > f2;
    typedef tbb::flow::composite_node< tbb::flow::tuple< int >, tbb::flow::tuple< int > > base_type;

public:
    tiny_node(tbb::flow::graph &g, bool hidden = false) : base_type(g), f1(g, tbb::flow::unlimited, passthru_body() ), f2(g, tbb::flow::unlimited, passthru_body() ) {
        tbb::flow::make_edge( f1, f2 );

        tbb::flow::tuple<tbb::flow::function_node< int, int >& > input_tuple(f1);
        tbb::flow::tuple<tbb::flow::function_node< int, int >& > output_tuple(f2);
        base_type::set_external_ports( input_tuple, output_tuple );

        if(hidden)
            base_type::add_nodes(f1, f2);
        else
            base_type::add_visible_nodes(f1, f2);

    }
};

int test_tiny(bool hidden = false) {
    tbb::flow::graph g;
    tbb::flow::function_node< int, int > f0( g, tbb::flow::unlimited, passthru_body() );
    tiny_node t(g, hidden);
    ASSERT(&tbb::flow::input_port<0>(t) == &t.f1, "f1 not bound to input port 0 in composite_node t");
    ASSERT(&tbb::flow::output_port<0>(t) == &t.f2, "f2 not bound to output port 0 in composite_node t");

    tiny_node t1(g, hidden);
    ASSERT(&tbb::flow::get<0>(t1.input_ports()) == &t1.f1, "f1 not bound to input port 0 in composite_node t1");
    ASSERT(&tbb::flow::get<0>(t1.output_ports()) == &t1.f2, "f2 not bound to output port 0 in composite_node t1");

    test_input_ports_return_ref(t1);
    test_output_ports_return_ref(t1);

    tiny_node t2(g, hidden);
    ASSERT(&tbb::flow::input_port<0>(t2) == &t2.f1, "f1 not bound to input port 0 in composite_node t2");
    ASSERT(&tbb::flow::output_port<0>(t2) == &t2.f2, "f2 not bound to output port 0 in composite_node t2");

    tbb::flow::function_node< int, int > f3( g, tbb::flow::unlimited, passthru_body() );
    tbb::flow::make_edge( f0, t );
    tbb::flow::make_edge( t, t1 );
    tbb::flow::make_edge( t1, t2 );
    tbb::flow::make_edge( t2 , f3 );
    tbb::flow::queue_node<int> q(g);
    tbb::flow::make_edge(f3, q);
    f0.try_put(1);
    g.wait_for_all();

    int i, j =0;
    q.try_get(i);
    ASSERT( i == 1, "item did not go through graph");
    q.try_get(j);
    ASSERT( !j, "unexpected item in graph");
    g.wait_for_all();

    tbb::flow::remove_edge(f3, q);
    tbb::flow::remove_edge(t2, f3);
    tbb::flow::remove_edge(t1, t2);

    tbb::flow::make_edge( t1 , f3 );
    tbb::flow::make_edge(f3, q);

    f0.try_put(2);
    g.wait_for_all();

    q.try_get(i);
    ASSERT( i == 2, "item did not go through graph after removal of edge");
    q.try_get(j);
    ASSERT( !j, "unexpected item in graph after removal of edge");

    return 0;
}

class adder_node : public tbb::flow::composite_node< tbb::flow::tuple< int, int >, tbb::flow::tuple< int > > {
public:
    tbb::flow::join_node< tbb::flow::tuple< int, int >, tbb::flow::queueing > j;
    tbb::flow::function_node< tbb::flow::tuple< int, int >, int > f;
private:
    typedef tbb::flow::composite_node< tbb::flow::tuple< int, int >, tbb::flow::tuple< int > > base_type;

    struct f_body {
        int operator()( const tbb::flow::tuple< int, int > &t ) {
            return tbb::flow::get<0>(t) + tbb::flow::get<1>(t);
        }
    };

public:
    adder_node(tbb::flow::graph &g, bool hidden = false) : base_type(g), j(g), f(g, tbb::flow::unlimited, f_body() ) {
        tbb::flow::make_edge( j, f );

        base_type::set_external_ports(base_type::input_ports_type(tbb::flow::input_port<0>(j), tbb::flow::input_port<1>(j)), base_type::output_ports_type(f));

        if (hidden)
            base_type::add_nodes(j, f);
        else
            base_type::add_visible_nodes(j, f);

    }
};

struct square_body { int operator()(int v) { return v*v; } };
struct cube_body { int operator()(int v) { return v*v*v; } };
int adder_sum(int i) {
    return (int)(pow(3*pow(i,3) + pow(i, 2),2));
}
int test_adder(bool hidden = false) {
    tbb::flow::graph g;
    tbb::flow::function_node<int,int> s(g, tbb::flow::unlimited, square_body());
    tbb::flow::function_node<int,int> c(g, tbb::flow::unlimited, cube_body());
    tbb::flow::function_node<int,int> p(g, tbb::flow::unlimited, passthru_body());

    adder_node a0(g, hidden);
    ASSERT(&tbb::flow::input_port<0>(a0) == &tbb::flow::input_port<0>(a0.j), "input_port 0 of j not bound to input port 0 in composite_node a0");
    ASSERT(&tbb::flow::input_port<1>(a0) == &tbb::flow::input_port<1>(a0.j), "input_port 1 of j not bound to input port 1 in composite_node a0");
    ASSERT(&tbb::flow::output_port<0>(a0) == &a0.f, "f not bound to output port 0 in composite_node a0");

    adder_node a1(g, hidden);
    ASSERT(&tbb::flow::get<0>(a0.input_ports()) == &tbb::flow::input_port<0>(a0.j), "input_port 0 of j not bound to input port 0 in composite_node a1");
    ASSERT(&tbb::flow::get<1>(a0.input_ports()) == &tbb::flow::input_port<1>(a0.j), "input_port1 of j not bound to input port 1 in composite_node a1");
    ASSERT(&tbb::flow::get<0>(a0.output_ports()) == &a0.f, "f not bound to output port 0 in composite_node a1");

    adder_node a2(g, hidden);
    ASSERT(&tbb::flow::input_port<0>(a2) == &tbb::flow::input_port<0>(a2.j), "input_port 0 of j not bound to input port 0 in composite_node a2");
    ASSERT(&tbb::flow::input_port<1>(a2) == &tbb::flow::input_port<1>(a2.j), "input_port 1 of j not bound to input port 1 in composite_node a2");
    ASSERT(&tbb::flow::output_port<0>(a2) == &a2.f, "f not bound to output port 0 in composite_node a2");

    adder_node a3(g, hidden);
    ASSERT(&tbb::flow::get<0>(a3.input_ports()) == &tbb::flow::input_port<0>(a3.j), "input_port 0 of j not bound to input port 0 in composite_node a3");
    ASSERT(&tbb::flow::get<1>(a3.input_ports()) == &tbb::flow::input_port<1>(a3.j), "input_port1 of j not bound to input port 1 in composite_node a3");
    ASSERT(&tbb::flow::get<0>(a3.output_ports()) == &a3.f, "f not bound to output port 0 in composite_node a3");

    tbb::flow::function_node<int,int> s2(g, tbb::flow::unlimited, square_body());
    tbb::flow::queue_node<int> q(g);

    tbb::flow::make_edge( s, tbb::flow::input_port<0>(a0) );
    tbb::flow::make_edge( c, tbb::flow::input_port<1>(a0) );

    tbb::flow::make_edge( c, tbb::flow::input_port<0>(a1) );
    tbb::flow::make_edge( c, tbb::flow::input_port<1>(a1) );

    tbb::flow::make_edge( tbb::flow::output_port<0>(a0), tbb::flow::input_port<0>(a2) );
    tbb::flow::make_edge( tbb::flow::output_port<0>(a1), tbb::flow::input_port<1>(a2) );

    tbb::flow::make_edge( tbb::flow::output_port<0>(a2), s2 );
    tbb::flow::make_edge( s2, q );

    int sum_total=0;
    int result=0;
    for ( int i = 1; i < 4; ++i ) {
       s.try_put(i);
       c.try_put(i);
       sum_total += adder_sum(i);
    g.wait_for_all();
    }

    int j;
    for ( int i = 1; i < 4; ++i ) {
        q.try_get(j);
        result += j;
    }
    g.wait_for_all();
    ASSERT(result == sum_total, "the sum from the graph does not match the calculated value");

    tbb::flow::remove_edge(s2, q);
    tbb::flow::remove_edge( a2, s2 );
    tbb::flow::make_edge( a0, a3 );
    tbb::flow::make_edge( a1, tbb::flow::input_port<1>(a3) );
    tbb::flow::make_edge( a3, s2 );
    tbb::flow::make_edge( s2, q );

    sum_total=0;
    result=0;
    for ( int i = 10; i < 20; ++i ) {
       s.try_put(i);
       c.try_put(i);
       sum_total += adder_sum(i);
    g.wait_for_all();
    }

    for ( int i = 10; i < 20; ++i ) {
        q.try_get(j);
        result += j;
    }
    g.wait_for_all();
    ASSERT(result == sum_total, "the new sum after the replacement of the nodes does not match the calculated value");

    return 0;
}

/*
                                              outer composite node (outer_node)
                                     |-------------------------------------------------------------------|
                                     |                                                                   |
                                     |  |------------------|  |------------------|  |------------------| |
             |---------------------| |--| inner composite  | /| inner composite  | /| inner composite  | | |-------------------|
             |broadcast node(input)|/|  | node             |/ | node             |/ | node             |-+-| queue node(output)|
             |---------------------|\|  |(inner_node1)     |\ | (inner_node2)    |\ | (inner_node3)    | | |-------------------|
                                     |--|                  | \|                  | \|                  | |
                                     |  |------------------|  |------------------|  |------------------| |
                                     |                                                                   |
                                     |-------------------------------------------------------------------|

*/
int test_nested_adder(bool hidden=false) {
    tbb::flow::graph g;
    tbb::flow::composite_node<tbb::flow::tuple<int, int>, tbb::flow::tuple<int> > outer_node(g);
    typedef tbb::flow::composite_node<tbb::flow::tuple<int, int>, tbb::flow::tuple<int> > base_type;
    tbb::flow::broadcast_node<int> input(g);
    tbb::flow::queue_node<int> output(g);

    adder_node inner_node1(g, hidden);
    adder_node inner_node2(g, hidden);
    adder_node inner_node3(g, hidden);

    outer_node.set_external_ports(base_type::input_ports_type(tbb::flow::input_port<0>(inner_node1), tbb::flow::input_port<1>(inner_node1)), base_type::output_ports_type(tbb::flow::output_port<0>(inner_node3)));

    ASSERT(&tbb::flow::input_port<0>(outer_node) == &tbb::flow::input_port<0>(inner_node1), "input port 0 of inner_node1 not bound to input port 0 in outer_node");
    ASSERT(&tbb::flow::input_port<1>(outer_node) == &tbb::flow::input_port<1>(inner_node1), "input port 1 of inner_node1 not bound to input port 1 in outer_node");
    ASSERT(&tbb::flow::output_port<0>(outer_node) == &tbb::flow::output_port<0>(inner_node3), "output port 0 of inner_node3 not bound to output port 0 in outer_node");

    tbb::flow::make_edge(input, tbb::flow::input_port<0>(outer_node)/*inner_node1*/);
    tbb::flow::make_edge(input, tbb::flow::input_port<1>(outer_node)/*inner_node1*/);

    tbb::flow::make_edge(inner_node1, tbb::flow::input_port<0>(inner_node2));
    tbb::flow::make_edge(inner_node1, tbb::flow::input_port<1>(inner_node2));

    tbb::flow::make_edge(inner_node2, tbb::flow::input_port<0>(inner_node3));
    tbb::flow::make_edge(inner_node2, tbb::flow::input_port<1>(inner_node3));

    tbb::flow::make_edge(outer_node/*inner_node3*/, output);

    if(hidden)
        outer_node.add_nodes(inner_node1, inner_node2, inner_node3);
    else
        outer_node.add_visible_nodes(inner_node1, inner_node2, inner_node3);

    int out;
    for (int i = 1; i < 200000; ++i) {
        input.try_put(i);
        g.wait_for_all();
        output.try_get(out);
        ASSERT(tbb::flow::output_port<0>(outer_node).try_get(out) == output.try_get(out), "output from outer_node does not match output from graph");
        ASSERT(out == 8*i, "output from outer_node not correct");
    }
    g.wait_for_all();

    return 0;
}

template< typename T >
class prefix_node : public tbb::flow::composite_node< tbb::flow::tuple< T, T, T, T, T >, tbb::flow::tuple< T, T, T, T, T > > {
    typedef tbb::flow::tuple< T, T, T, T, T > my_tuple_t;
public:
    tbb::flow::join_node< my_tuple_t, tbb::flow::queueing > j;
    tbb::flow::split_node< my_tuple_t > s;
private:
    tbb::flow::function_node< my_tuple_t, my_tuple_t > f;
    typedef tbb::flow::composite_node< my_tuple_t, my_tuple_t > base_type;

    struct f_body {
        my_tuple_t operator()( const my_tuple_t &t ) {
            return my_tuple_t( tbb::flow::get<0>(t),
                               tbb::flow::get<0>(t) + tbb::flow::get<1>(t),
                               tbb::flow::get<0>(t) + tbb::flow::get<1>(t) + tbb::flow::get<2>(t),
                               tbb::flow::get<0>(t) + tbb::flow::get<1>(t) + tbb::flow::get<2>(t) + tbb::flow::get<3>(t),
                               tbb::flow::get<0>(t) + tbb::flow::get<1>(t) + tbb::flow::get<2>(t) + tbb::flow::get<3>(t) + tbb::flow::get<4>(t) );
        }
    };

public:
    prefix_node(tbb::flow::graph &g, bool hidden = false ) : base_type(g), j(g), s(g), f(g, tbb::flow::serial, f_body() ) {
        tbb::flow::make_edge( j, f );
        tbb::flow::make_edge( f, s );

    typename base_type::input_ports_type input_tuple(tbb::flow::input_port<0>(j), tbb::flow::input_port<1>(j), tbb::flow::input_port<2>(j), tbb::flow::input_port<3>(j), tbb::flow::input_port<4>(j));

    typename base_type::output_ports_type output_tuple(tbb::flow::output_port<0>(s), tbb::flow::output_port<1>(s), tbb::flow::output_port<2>(s), tbb::flow::output_port<3>(s), tbb::flow::output_port<4>(s));

    base_type::set_external_ports(input_tuple, output_tuple);

        if(hidden)
            base_type::add_nodes(j,s,f);
        else
            base_type::add_visible_nodes(j,s,f);

    }
};

int test_prefix(bool hidden = false) {
    tbb::flow::graph g;
    prefix_node<double> p(g, hidden);

    ASSERT(&tbb::flow::get<0>(p.input_ports()) == &tbb::flow::input_port<0>(p.j), "input port 0 of j is not bound to input port 0 of composite node p");
    ASSERT(&tbb::flow::input_port<1>(p.j) == &tbb::flow::input_port<1>(p.j), "input port 1 of j is not bound to input port 1 of composite node p");
    ASSERT(&tbb::flow::get<2>(p.input_ports()) == &tbb::flow::input_port<2>(p.j), "input port 2 of j is not bound to input port 2 of composite node p");
    ASSERT(&tbb::flow::input_port<3>(p.j) == &tbb::flow::input_port<3>(p.j), "input port 3 of j is not bound to input port 3 of composite node p");
    ASSERT(&tbb::flow::get<4>(p.input_ports()) == &tbb::flow::input_port<4>(p.j), "input port 4 of j is not bound to input port 4 of composite node p");


    ASSERT(&tbb::flow::get<0>(p.output_ports()) == &tbb::flow::output_port<0>(p.s), "output port 0 of s is not bound to output port 0 of composite node p");
    ASSERT(&tbb::flow::output_port<1>(p.s) == &tbb::flow::output_port<1>(p.s), "output port 1 of s is not bound to output port 1 of composite node p");
    ASSERT(&tbb::flow::get<2>(p.output_ports()) == &tbb::flow::output_port<2>(p.s), "output port 2 of s is not bound to output port 2 of composite node p");
    ASSERT(&tbb::flow::output_port<3>(p.s) == &tbb::flow::output_port<3>(p.s), "output port 3 of s is not bound to output port 3 of composite node p");
    ASSERT(&tbb::flow::get<4>(p.output_ports()) == &tbb::flow::output_port<4>(p.s), "output port 4 of s is not bound to output port 4 of composite node p");

    std::vector< tbb::flow::queue_node<double> > v( 5, tbb::flow::queue_node<double>(g) );
    tbb::flow::make_edge( tbb::flow::output_port<0>(p), v[0] );
    tbb::flow::make_edge( tbb::flow::output_port<1>(p), v[1] );
    tbb::flow::make_edge( tbb::flow::output_port<2>(p), v[2] );
    tbb::flow::make_edge( tbb::flow::output_port<3>(p), v[3] );
    tbb::flow::make_edge( tbb::flow::output_port<4>(p), v[4] );

    for(  double offset = 1; offset < 10000; offset *= 10 ) {
        tbb::flow::input_port<0>(p).try_put( offset );
        tbb::flow::input_port<1>(p).try_put( offset + 1 );
        tbb::flow::input_port<2>(p).try_put( offset + 2 );
        tbb::flow::input_port<3>(p).try_put( offset + 3 );
        tbb::flow::input_port<4>(p).try_put( offset + 4 );
    }
    g.wait_for_all();

    double x;
    while ( v[0].try_get(x) ) {
        g.wait_for_all();
        for ( int i = 1; i < 5; ++i ) {
            v[i].try_get(x);
            g.wait_for_all();
        }
    }
    return 0;
}

void input_only_output_only_composite(bool hidden) {
    tbb::flow::graph g;
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    tbb::flow::composite_node<tbb::flow::tuple<int>, tbb::flow::tuple<int> > input_output(g, "test_name");
#else
    tbb::flow::composite_node<tbb::flow::tuple<int>, tbb::flow::tuple<int> > input_output(g);
#endif
    typedef tbb::flow::composite_node<tbb::flow::tuple<int>, tbb::flow::tuple<> > input_only_composite;
    typedef tbb::flow::composite_node<tbb::flow::tuple<>, tbb::flow::tuple<int> > output_only_composite;
    typedef tbb::flow::source_node<int> src_type;
    typedef tbb::flow::queue_node<int> q_type;
    typedef tbb::flow::function_node<int, int> f_type;

    int num = 0;
    int finish=1000;
    int step = 4;

    input_only_composite a_in(g);
    output_only_composite a_out(g);

    src_type src(g, src_body(finish, step), false);
    q_type que(g);
    f_type f(g, 1, passthru_body());

    tbb::flow::tuple<f_type& > input_tuple(f);
    a_in.set_external_ports(input_tuple);
    ASSERT(&tbb::flow::get<0>(a_in.input_ports()) == &f, "f not bound to input port 0 in composite_node a_in");

    tbb::flow::tuple<src_type&> output_tuple(src);
    a_out.set_external_ports(output_tuple);
    ASSERT(&tbb::flow::get<0>(a_out.output_ports()) == &src, "src not bound to output port 0 in composite_node a_out");

    if(hidden) {
        a_in.add_nodes(f, que);
        a_out.add_nodes(src);
    } else {
        a_in.add_visible_nodes(f, que);
        a_out.add_visible_nodes(src);
    }

    tbb::flow::make_edge(a_out, a_in);
    tbb::flow::make_edge(f, que);
    src.activate();
    g.wait_for_all();

    for(int i = 1; i<finish/step; ++i) {
        que.try_get(num);
        ASSERT(num == 4*i - 3, "number does not match position in sequence");
    }
    g.wait_for_all();
}

#endif // __TBB_FLOW_GRAPH_CPP11_FEATURES

int TestMain() {

#if __TBB_FLOW_GRAPH_CPP11_FEATURES

    add_all_nodes();
    test_tiny(false);
    test_tiny(true);
    test_adder(false);
    test_adder(true);
    test_nested_adder(true);
    test_nested_adder(false);
    test_prefix(false);
    test_prefix(true);
    input_only_output_only_composite(true);
    input_only_output_only_composite(false);

    return Harness::Done;
#else
    return Harness::Skipped;
#endif

}
