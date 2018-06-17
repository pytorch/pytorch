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

//
// Tests
//

#if defined(_MSC_VER) && _MSC_VER < 1600
    #pragma warning (disable : 4503) //disabling the "decorated name length exceeded" warning for VS2008 and earlier
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
template< typename T >
class test_indexer_extract {
protected:
    typedef tbb::flow::indexer_node<T, T> my_node_t;
    typedef tbb::flow::queue_node<T> in_node_t;
    typedef tbb::flow::queue_node<typename my_node_t::output_type> out_node_t;

    tbb::flow::graph g;
    in_node_t in0;
    in_node_t in1;
    in_node_t in2;
    my_node_t middle;
    out_node_t out0;
    out_node_t out1;
    in_node_t *ins[3];
    out_node_t *outs[2];
    typename in_node_t::successor_type *ms_p0_ptr;
    typename in_node_t::successor_type *ms_p1_ptr;
    typename out_node_t::predecessor_type *mp_ptr;
    typename in_node_t::predecessor_list_type in0_p_list;
    typename in_node_t::successor_list_type in0_s_list;
    typename in_node_t::predecessor_list_type in1_p_list;
    typename in_node_t::successor_list_type in1_s_list;
    typename in_node_t::predecessor_list_type in2_p_list;
    typename in_node_t::successor_list_type in2_s_list;
    typename out_node_t::predecessor_list_type out0_p_list;
    typename out_node_t::successor_list_type out0_s_list;
    typename out_node_t::predecessor_list_type out1_p_list;
    typename out_node_t::successor_list_type out1_s_list;
    typename in_node_t::predecessor_list_type mp0_list;
    typename in_node_t::predecessor_list_type mp1_list;
    typename out_node_t::successor_list_type ms_list;

    virtual void set_up_lists() {
        in0_p_list.clear();
        in0_s_list.clear();
        in1_p_list.clear();
        in1_s_list.clear();
        in2_p_list.clear();
        in2_s_list.clear();
        out0_p_list.clear();
        out0_s_list.clear();
        out1_p_list.clear();
        out1_s_list.clear();
        mp0_list.clear();
        mp1_list.clear();
        ms_list.clear();

        in0.copy_predecessors(in0_p_list);
        in0.copy_successors(in0_s_list);
        in1.copy_predecessors(in1_p_list);
        in1.copy_successors(in1_s_list);
        in2.copy_predecessors(in2_p_list);
        in2.copy_successors(in2_s_list);
        tbb::flow::input_port<0>(middle).copy_predecessors(mp0_list);
        tbb::flow::input_port<1>(middle).copy_predecessors(mp1_list);
        middle.copy_successors(ms_list);
        out0.copy_predecessors(out0_p_list);
        out0.copy_successors(out0_s_list);
        out1.copy_predecessors(out1_p_list);
        out1.copy_successors(out1_s_list);
    }

    void check_output(int &r, typename my_node_t::output_type &v) {
        T t = tbb::flow::cast_to<T>(v);
        if ( t == 1 || t == 2 ) {
            ASSERT( v.tag() == 0, "value came in on wrong port" );
        } else if ( t == 4 || t == 8 ) {
            ASSERT( v.tag() == 1, "value came in on wrong port" );
        } else {
            ASSERT( false, "incorrect value passed through indexer_node" );
        }
        ASSERT( (r&t) == 0, "duplicate value passed through indexer_node" );
        r |= t;
    }

    void make_and_validate_full_graph() {
        /*     in0                         */
        /*         \                       */
        /*           port0          out0   */
        /*         /       |      /        */
        /*     in1         middle          */
        /*                 |      \        */
        /*     in2 - port1          out1   */
        tbb::flow::make_edge( in0, tbb::flow::input_port<0>(middle) );
        tbb::flow::make_edge( in1, tbb::flow::input_port<0>(middle) );
        tbb::flow::make_edge( in2, tbb::flow::input_port<1>(middle) );
        tbb::flow::make_edge( middle, out0 );
        tbb::flow::make_edge( middle, out1 );

        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 1 && in0_s_list.size() == 1 && *(in0_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
        ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in2.successor_count() == 1 && in2_s_list.size() == 1 && *(in2_s_list.begin()) == ms_p1_ptr, "expected 1 successor" );
        ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 2 && mp0_list.size() == 2, "expected 2 predecessors" );
        ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 1 && mp1_list.size() == 1, "expected 1 predecessors" );
        ASSERT( middle.successor_count() == 2 && ms_list.size() == 2, "expected 2 successors" );
        ASSERT( out0.predecessor_count() == 1 && out0_p_list.size() == 1 && *(out0_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        int first_pred = *(mp0_list.begin()) == ins[0] ? 0 : ( *(mp0_list.begin()) == ins[1] ? 1 : -1 );
        typename in_node_t::predecessor_list_type::iterator piv = mp0_list.begin();++piv;
        int second_pred = *piv == ins[0] ? 0 : ( *piv == ins[1] ? 1 : -1 );
        ASSERT( first_pred != -1 && second_pred != -1 && first_pred != second_pred, "bad predecessor(s) for middle port 0" );

        ASSERT( *(mp1_list.begin()) == ins[2], "bad predecessor for middle port 1" );

        int first_succ = *(ms_list.begin()) == outs[0] ? 0 : ( *(ms_list.begin()) == outs[1] ? 1 : -1 );
        typename out_node_t::successor_list_type::iterator ms_vec_iter = ms_list.begin(); ++ms_vec_iter;
        int second_succ = *ms_vec_iter == outs[0] ? 0 : ( *ms_vec_iter == outs[1] ? 1 : -1 );
        ASSERT( first_succ != -1 && second_succ != -1 && first_succ != second_succ, "bad successor(s) for middle" );

        in0.try_put(1);
        in1.try_put(2);
        in2.try_put(8);
        in2.try_put(4);
        g.wait_for_all();

        T v_in;

        ASSERT( in0.try_get(v_in) == false, "buffer should not have a value" );
        ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
        ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
        ASSERT( in2.try_get(v_in) == false, "buffer should not have a value" );
        ASSERT( in2.try_get(v_in) == false, "buffer should not have a value" );

        typename my_node_t::output_type v;
        T r = 0;
        while ( out0.try_get(v) ) {
            check_output(r,v);
            g.wait_for_all();
        }
        ASSERT( r == 15, "not all values received" );

        r = 0;
        while ( out1.try_get(v) ) {
            check_output(r,v);
            g.wait_for_all();
        }
        ASSERT( r == 15, "not all values received" );
        g.wait_for_all();
    }

    void validate_partial_graph() {
        /*     in0                         */
        /*                                 */
        /*           port0          out0   */
        /*         /       |               */
        /*     in1         middle          */
        /*                 |      \        */
        /*     in2 - port1          out1   */
        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
        ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in2.successor_count() == 1 && in2_s_list.size() == 1 && *(in2_s_list.begin()) == ms_p1_ptr, "expected 1 successor" );
        ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 1 && mp0_list.size() == 1 && *(mp0_list.begin()) == ins[1], "expected 1 predecessor" );
        ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 1 && mp1_list.size() == 1 && *(mp1_list.begin()) == ins[2], "expected 1 predecessor" );
        ASSERT( middle.successor_count() == 1 && ms_list.size() == 1 && *(ms_list.begin()) == outs[1], "expected 1 successor" );
        ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        in0.try_put(1);
        in1.try_put(2);
        in2.try_put(8);
        in2.try_put(4);
        g.wait_for_all();

        T v_in;
        typename my_node_t::output_type v;

        ASSERT( in0.try_get(v_in) == true && v_in == 1, "buffer should have a value of 1" );
        ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
        ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
        ASSERT( in0.try_get(v_in) == false, "buffer should not have a value" );

        T r = 0;
        while ( out1.try_get(v) ) {
            check_output(r,v);
            g.wait_for_all();
        }
        ASSERT( r == 14, "not all values received" );
        g.wait_for_all();
    }

    void validate_empty_graph() {
        /*     in0                         */
        /*                                 */
        /*            port0         out0   */
        /*                |                */
        /*     in1         middle          */
        /*                 |               */
        /*     in2   port1          out1   */
        set_up_lists();

        ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in1.successor_count() == 0 && in1_s_list.size() == 0, "expected 0 successors" );
        ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( in2.successor_count() == 0 && in2_s_list.size() == 0, "expected 0 successors" );
        ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 0 && mp0_list.size() == 0, "expected 0 predecessors" );
        ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 0 && mp1_list.size() == 0, "expected 0 predecessors" );
        ASSERT( middle.successor_count() == 0 && ms_list.size() == 0, "expected 0 successors" );
        ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
        ASSERT( out1.predecessor_count() == 0 && out1_p_list.size() == 0, "expected 0 predecessors" );
        ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

        in0.try_put(1);
        in1.try_put(2);
        in2.try_put(8);
        in2.try_put(4);
        g.wait_for_all();

        T v_in;
        typename my_node_t::output_type v;

        ASSERT( in0.try_get(v_in) == true && v_in == 1, "buffer should have a value of 1" );
        ASSERT( in1.try_get(v_in) == true && v_in == 2, "buffer should have a value of 2" );
        ASSERT( in2.try_get(v_in) == true && v_in == 8, "buffer should have a value of 8" );
        ASSERT( in2.try_get(v_in) == true && v_in == 4, "buffer should have a value of 4" );
        ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
        ASSERT( out1.try_get(v) == false, "buffer should not have a value" );
        g.wait_for_all();
        g.reset(); // NOTE: this should not be necessary!!!!!  But it is!!!!
    }

public:

    test_indexer_extract() : in0(g), in1(g), in2(g), middle(g), out0(g), out1(g) {
        ins[0] = &in0;
        ins[1] = &in1;
        ins[2] = &in2;
        outs[0] = &out0;
        outs[1] = &out1;
        ms_p0_ptr = static_cast< typename in_node_t::successor_type * >(&tbb::flow::input_port<0>(middle));
        ms_p1_ptr = static_cast< typename in_node_t::successor_type * >(&tbb::flow::input_port<1>(middle));
        mp_ptr = static_cast< typename out_node_t::predecessor_type *>(&middle);
    }

    virtual ~test_indexer_extract() {}

    void run_tests() {
        REMARK("full graph\n");
        make_and_validate_full_graph();

        in0.extract();
        out0.extract();
        REMARK("partial graph\n");
        validate_partial_graph();

        in1.extract();
        in2.extract();
        out1.extract();
        REMARK("empty graph\n");
        validate_empty_graph();

        REMARK("full graph\n");
        make_and_validate_full_graph();

        middle.extract();
        REMARK("empty graph\n");
        validate_empty_graph();

        REMARK("full graph\n");
        make_and_validate_full_graph();

        in0.extract();
        in1.extract();
        in2.extract();
        middle.extract();
        REMARK("empty graph\n");
        validate_empty_graph();

        REMARK("full graph\n");
        make_and_validate_full_graph();

        out0.extract();
        out1.extract();
        middle.extract();
        REMARK("empty graph\n");
        validate_empty_graph();

        REMARK("full graph\n");
        make_and_validate_full_graph();
    }
};
#endif

const int Count = 150;
const int MaxPorts = 10;
const int MaxNSources = 5; // max # of source_nodes to register for each indexer_node input in parallel test
bool outputCheck[MaxPorts][Count];  // for checking output

void
check_outputCheck( int nUsed, int maxCnt) {
    for(int i=0; i < nUsed; ++i) {
        for( int j = 0; j < maxCnt; ++j) {
            ASSERT(outputCheck[i][j], NULL);
        }
    }
}

void
reset_outputCheck( int nUsed, int maxCnt) {
    for(int i=0; i < nUsed; ++i) {
        for( int j = 0; j < maxCnt; ++j) {
            outputCheck[i][j] = false;
        }
    }
}

class test_class {
    public:
        test_class() { my_val = 0; }
        test_class(int i) { my_val = i; }
        operator int() { return my_val; }
    private:
        int my_val;
};

template<typename T>
class name_of {
public:
    static const char* name() { return  "Unknown"; }
};
template<>
class name_of<int> {
public:
    static const char* name() { return  "int"; }
};
template<>
class name_of<float> {
public:
    static const char* name() { return  "float"; }
};
template<>
class name_of<double> {
public:
    static const char* name() { return  "double"; }
};
template<>
class name_of<long> {
public:
    static const char* name() { return  "long"; }
};
template<>
class name_of<short> {
public:
    static const char* name() { return  "short"; }
};
template<>
class name_of<test_class> {
public:
    static const char* name() { return  "test_class"; }
};

// TT must be arithmetic, and shouldn't wrap around for reasonable sizes of Count (which is now 150, and maxPorts is 10,
// so the max number generated right now is 1500 or so.)  Source will generate a series of TT with value
// (init_val + (i-1)*addend) * my_mult, where i is the i-th invocation of the body.  We are attaching addend
// source nodes to a indexer_port, and each will generate part of the numerical series the port is expecting
// to receive.  If there is only one source node, the series order will be maintained; if more than one,
// this is not guaranteed.
// The manual specifies bodies can be assigned, so we can't hide the operator=.
template<typename TT>
class source_body {
    TT my_mult;
    int my_count;
    int addend;
public:
    source_body(TT multiplier, int init_val, int addto) : my_mult(multiplier), my_count(init_val), addend(addto) { }
    bool operator()( TT &v) {
        int lc = my_count;
        v = my_mult * (TT)my_count;
        my_count += addend;
        return lc < Count;
    }
};

// allocator for indexer_node.

template<typename IType>
class makeIndexer {
public:
    static IType *create() {
        IType *temp = new IType();
        return temp;
    }
    static void destroy(IType *p) { delete p; }
};

template<int ELEM, typename INT>
struct getval_helper {

    typedef typename INT::output_type OT;
    typedef typename tbb::flow::tuple_element<ELEM-1, typename INT::tuple_types>::type stored_type;

    static int get_integer_val(OT const &o) {
        stored_type res = tbb::flow::cast_to<stored_type>(o);
        return (int)res;
    }
};

// holder for source_node pointers for eventual deletion

static void* all_source_nodes[MaxPorts][MaxNSources];

template<int ELEM, typename INT>
class source_node_helper {
public:
    typedef INT indexer_node_type;
    typedef typename indexer_node_type::output_type TT;
    typedef typename tbb::flow::tuple_element<ELEM-1,typename INT::tuple_types>::type IT;
    typedef typename tbb::flow::source_node<IT> my_source_node_type;
    static void print_remark() {
        source_node_helper<ELEM-1,INT>::print_remark();
        REMARK(", %s", name_of<IT>::name());
    }
    static void add_source_nodes(indexer_node_type &my_indexer, tbb::flow::graph &g, int nInputs) {
        for(int i=0; i < nInputs; ++i) {
            my_source_node_type *new_node = new my_source_node_type(g, source_body<IT>((IT)(ELEM+1), i, nInputs));
            tbb::flow::make_edge(*new_node, tbb::flow::input_port<ELEM-1>(my_indexer));
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            ASSERT(new_node->successor_count() == 1, NULL);
#endif
            all_source_nodes[ELEM-1][i] = (void *)new_node;
        }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        ASSERT(tbb::flow::input_port<ELEM-1>(my_indexer).predecessor_count() == (size_t)nInputs, NULL);
#endif
        // add the next source_node
        source_node_helper<ELEM-1, INT>::add_source_nodes(my_indexer, g, nInputs);
    }
    static void check_value(TT &v) {
        if(v.tag() == ELEM-1) {
            int ival = getval_helper<ELEM,INT>::get_integer_val(v);
            ASSERT(!(ival%(ELEM+1)), NULL);
            ival /= (ELEM+1);
            ASSERT(!outputCheck[ELEM-1][ival], NULL);
            outputCheck[ELEM-1][ival] = true;
        }
        else {
            source_node_helper<ELEM-1,INT>::check_value(v);
        }
    }

    static void remove_source_nodes(indexer_node_type& my_indexer, int nInputs) {
        for(int i=0; i< nInputs; ++i) {
            my_source_node_type *dp = reinterpret_cast<my_source_node_type *>(all_source_nodes[ELEM-1][i]);
            tbb::flow::remove_edge(*dp, tbb::flow::input_port<ELEM-1>(my_indexer));
            delete dp;
        }
        source_node_helper<ELEM-1, INT>::remove_source_nodes(my_indexer, nInputs);
    }
};

template<typename INT>
class source_node_helper<1, INT> {
    typedef INT indexer_node_type;
    typedef typename indexer_node_type::output_type TT;
    typedef typename tbb::flow::tuple_element<0, typename INT::tuple_types>::type IT;
    typedef typename tbb::flow::source_node<IT> my_source_node_type;
public:
    static void print_remark() {
        REMARK("Parallel test of indexer_node< %s", name_of<IT>::name());
    }
    static void add_source_nodes(indexer_node_type &my_indexer, tbb::flow::graph &g, int nInputs) {
        for(int i=0; i < nInputs; ++i) {
            my_source_node_type *new_node = new my_source_node_type(g, source_body<IT>((IT)2, i, nInputs));
            tbb::flow::make_edge(*new_node, tbb::flow::input_port<0>(my_indexer));
            all_source_nodes[0][i] = (void *)new_node;
        }
    }
    static void check_value(TT &v) {
        int ival = getval_helper<1,INT>::get_integer_val(v);
        ASSERT(!(ival%2), NULL);
        ival /= 2;
        ASSERT(!outputCheck[0][ival], NULL);
        outputCheck[0][ival] = true;
    }
    static void remove_source_nodes(indexer_node_type& my_indexer, int nInputs) {
        for(int i=0; i < nInputs; ++i) {
            my_source_node_type *dp = reinterpret_cast<my_source_node_type *>(all_source_nodes[0][i]);
            tbb::flow::remove_edge(*dp, tbb::flow::input_port<0>(my_indexer));
            delete dp;
        }
    }
};

template<typename IType>
class parallel_test {
public:
    typedef typename IType::output_type TType;
    typedef typename IType::tuple_types union_types;
    static const int SIZE = tbb::flow::tuple_size<union_types>::value;
    static void test() {
        TType v;
        source_node_helper<SIZE,IType>::print_remark();
        REMARK(" >\n");
        for(int i=0; i < MaxPorts; ++i) {
            for(int j=0; j < MaxNSources; ++j) {
                all_source_nodes[i][j] = NULL;
            }
        }
        for(int nInputs = 1; nInputs <= MaxNSources; ++nInputs) {
            tbb::flow::graph g;
            IType* my_indexer = new IType(g); //makeIndexer<IType>::create();
            tbb::flow::queue_node<TType> outq1(g);
            tbb::flow::queue_node<TType> outq2(g);

            tbb::flow::make_edge(*my_indexer, outq1);
            tbb::flow::make_edge(*my_indexer, outq2);

            source_node_helper<SIZE, IType>::add_source_nodes((*my_indexer), g, nInputs);

            g.wait_for_all();

            reset_outputCheck(SIZE, Count);
            for(int i=0; i < Count*SIZE; ++i) {
                ASSERT(outq1.try_get(v), NULL);
                source_node_helper<SIZE, IType>::check_value(v);
            }

            check_outputCheck(SIZE, Count);
            reset_outputCheck(SIZE, Count);

            for(int i=0; i < Count*SIZE; i++) {
                ASSERT(outq2.try_get(v), NULL);;
                source_node_helper<SIZE, IType>::check_value(v);
            }
            check_outputCheck(SIZE, Count);

            ASSERT(!outq1.try_get(v), NULL);
            ASSERT(!outq2.try_get(v), NULL);

            source_node_helper<SIZE, IType>::remove_source_nodes((*my_indexer), nInputs);
            tbb::flow::remove_edge(*my_indexer, outq1);
            tbb::flow::remove_edge(*my_indexer, outq2);
            makeIndexer<IType>::destroy(my_indexer);
        }
    }
};

std::vector<int> last_index_seen;

template<int ELEM, typename IType>
class serial_queue_helper {
public:
    typedef typename IType::output_type OT;
    typedef typename IType::tuple_types TT;
    typedef typename tbb::flow::tuple_element<ELEM-1,TT>::type IT;
    static void print_remark() {
        serial_queue_helper<ELEM-1,IType>::print_remark();
        REMARK(", %s", name_of<IT>::name());
    }
    static void fill_one_queue(int maxVal, IType &my_indexer) {
        // fill queue to "left" of me
        serial_queue_helper<ELEM-1,IType>::fill_one_queue(maxVal,my_indexer);
        for(int i = 0; i < maxVal; ++i) {
            ASSERT(tbb::flow::input_port<ELEM-1>(my_indexer).try_put((IT)(i*(ELEM+1))), NULL);
        }
    }
    static void put_one_queue_val(int myVal, IType &my_indexer) {
        // put this val to my "left".
        serial_queue_helper<ELEM-1,IType>::put_one_queue_val(myVal, my_indexer);
        ASSERT(tbb::flow::input_port<ELEM-1>(my_indexer).try_put((IT)(myVal*(ELEM+1))), NULL);
    }
    static void check_queue_value(OT &v) {
        if(ELEM - 1 == v.tag()) {
            // this assumes each or node input is queueing.
            int rval = getval_helper<ELEM,IType>::get_integer_val(v);
            ASSERT( rval == (last_index_seen[ELEM-1]+1)*(ELEM+1), NULL);
            last_index_seen[ELEM-1] = rval / (ELEM+1);
        }
        else {
            serial_queue_helper<ELEM-1,IType>::check_queue_value(v);
        }
    }
};

template<typename IType>
class serial_queue_helper<1, IType> {
public:
    typedef typename IType::output_type OT;
    typedef typename IType::tuple_types TT;
    typedef typename tbb::flow::tuple_element<0,TT>::type IT;
    static void print_remark() {
        REMARK("Serial test of indexer_node< %s", name_of<IT>::name());
    }
    static void fill_one_queue(int maxVal, IType &my_indexer) {
        for(int i = 0; i < maxVal; ++i) {
            ASSERT(tbb::flow::input_port<0>(my_indexer).try_put((IT)(i*2)), NULL);
        }
    }
    static void put_one_queue_val(int myVal, IType &my_indexer) {
        ASSERT(tbb::flow::input_port<0>(my_indexer).try_put((IT)(myVal*2)), NULL);
    }
    static void check_queue_value(OT &v) {
        ASSERT(v.tag() == 0, NULL);  // won't get here unless true
        int rval = getval_helper<1,IType>::get_integer_val(v);
        ASSERT( rval == (last_index_seen[0]+1)*2, NULL);
        last_index_seen[0] = rval / 2;
    }
};

template<typename IType, typename TType, int SIZE>
void test_one_serial( IType &my_indexer, tbb::flow::graph &g) {
    last_index_seen.clear();
    for(int ii=0; ii < SIZE; ++ii) last_index_seen.push_back(-1);

    typedef TType q3_input_type;
    tbb::flow::queue_node< q3_input_type >  q3(g);
    q3_input_type v;

    tbb::flow::make_edge(my_indexer, q3);
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    ASSERT(my_indexer.successor_count() == 1, NULL);
    ASSERT(tbb::flow::input_port<0>(my_indexer).predecessor_count() == 0, NULL);
#endif

    // fill each queue with its value one-at-a-time
    for (int i = 0; i < Count; ++i ) {
        serial_queue_helper<SIZE,IType>::put_one_queue_val(i,my_indexer);
    }

    g.wait_for_all();
    for (int i = 0; i < Count * SIZE; ++i ) {
        g.wait_for_all();
        ASSERT(q3.try_get( v ), "Error in try_get()");
        {
            serial_queue_helper<SIZE,IType>::check_queue_value(v);
        }
    }
    ASSERT(!q3.try_get( v ), "extra values in output queue");
    for(int ii=0; ii < SIZE; ++ii) last_index_seen[ii] = -1;

    // fill each queue completely before filling the next.
    serial_queue_helper<SIZE, IType>::fill_one_queue(Count,my_indexer);

    g.wait_for_all();
    for (int i = 0; i < Count*SIZE; ++i ) {
        g.wait_for_all();
        ASSERT(q3.try_get( v ), "Error in try_get()");
        {
            serial_queue_helper<SIZE,IType>::check_queue_value(v);
        }
    }
    ASSERT(!q3.try_get( v ), "extra values in output queue");
}

//
// Single predecessor at each port, single accepting successor
//   * put to buffer before port0, then put to buffer before port1, ...
//   * fill buffer before port0 then fill buffer before port1, ...

template<typename IType>
class serial_test {
    typedef typename IType::output_type TType;  // this is the union
    typedef typename IType::tuple_types union_types;
    static const int SIZE = tbb::flow::tuple_size<union_types>::value;
public:
static void test() {
    tbb::flow::graph g;
    static const int ELEMS = 3;
    IType* my_indexer = new IType(g); //makeIndexer<IType>::create(g);

    test_input_ports_return_ref(*my_indexer);

    serial_queue_helper<SIZE, IType>::print_remark(); REMARK(" >\n");

    test_one_serial<IType,TType,SIZE>(*my_indexer, g);

    std::vector<IType> indexer_vector(ELEMS,*my_indexer);

    makeIndexer<IType>::destroy(my_indexer);

    for(int e = 0; e < ELEMS; ++e) {
        test_one_serial<IType,TType,SIZE>(indexer_vector[e], g);
    }
}

}; // serial_test

template<
      template<typename> class TestType,  // serial_test or parallel_test
      typename T0, typename T1=void, typename T2=void, typename T3=void, typename T4=void,
      typename T5=void, typename T6=void, typename T7=void, typename T8=void, typename T9=void> // type of the inputs to the indexer_node
class generate_test {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

//specializations for indexer node inputs
template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3, typename T4,
      typename T5, typename T6, typename T7, typename T8>
class generate_test<TestType, T0, T1, T2, T3, T4, T5, T6, T7, T8> {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4, T5, T6, T7, T8>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3, typename T4,
      typename T5, typename T6, typename T7>
class generate_test<TestType, T0, T1, T2, T3, T4, T5, T6, T7> {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4, T5, T6, T7>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3, typename T4,
      typename T5, typename T6>
class generate_test<TestType, T0, T1, T2, T3, T4, T5, T6> {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4, T5, T6>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3, typename T4,
      typename T5>
class generate_test<TestType, T0, T1, T2, T3, T4, T5>  {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4, T5>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3, typename T4>
class generate_test<TestType, T0, T1, T2, T3, T4>  {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3, T4>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2, typename T3>
class generate_test<TestType, T0, T1, T2, T3> {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2, T3>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1, typename T2>
class generate_test<TestType, T0, T1, T2> {
public:
    typedef tbb::flow::indexer_node<T0, T1, T2>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0, typename T1>
class generate_test<TestType, T0, T1> {
public:
    typedef tbb::flow::indexer_node<T0, T1>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

template<
      template<typename> class TestType,
      typename T0>
class generate_test<TestType, T0> {
public:
    typedef tbb::flow::indexer_node<T0>  indexer_node_type;
    static void do_test() {
        TestType<indexer_node_type>::test();
    }
};

int TestMain() {
    REMARK("Testing indexer_node, ");
#if __TBB_USE_TBB_TUPLE
    REMARK("using TBB tuple\n");
#else
    REMARK("using platform tuple\n");
#endif

   for (int p = 0; p < 2; ++p) {
       generate_test<serial_test, float>::do_test();
#if MAX_TUPLE_TEST_SIZE >= 4
       generate_test<serial_test, float, double, int>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
       generate_test<serial_test, double, double, int, long, int, short>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 8
       generate_test<serial_test, float, double, double, double, float, int, float, long>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 10
       generate_test<serial_test, float, double, int, double, double, float, long, int, float, long>::do_test();
#endif
       generate_test<parallel_test, float, double>::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
       generate_test<parallel_test, float, int, long>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 5
       generate_test<parallel_test, double, double, int, int, short>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
       generate_test<parallel_test, float, int, double, float, long, float, long>::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 9
       generate_test<parallel_test, float, double, int, double, double, long, int, float, long>::do_test();
#endif
   }
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
   test_indexer_extract<int>().run_tests();
#endif
   return Harness::Done;
}
