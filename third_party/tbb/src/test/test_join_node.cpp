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

#include "test_join_node.h"

static tbb::atomic<int> output_count;

// get the tag from the output tuple and emit it.
// the first tuple component is tag * 2 cast to the type
template<typename OutputTupleType>
class recirc_output_func_body {
public:
    // we only need this to use source_node_helper
    typedef typename tbb::flow::join_node<OutputTupleType, tbb::flow::tag_matching> join_node_type;
    static const int N = tbb::flow::tuple_size<OutputTupleType>::value;
    int operator()(const OutputTupleType &v) {
        int out = int(tbb::flow::get<0>(v))/2;
        source_node_helper<N, join_node_type>::only_check_value(out, v);
        ++output_count;
        return out;
    }
};

template<typename JType>
class tag_recirculation_test {
public:
    typedef typename JType::output_type TType;
    typedef typename tbb::flow::tuple<int, tbb::flow::continue_msg> input_tuple_type;
    typedef tbb::flow::join_node<input_tuple_type, tbb::flow::reserving> input_join_type;
    static const int N = tbb::flow::tuple_size<TType>::value;
    static void test() {
        source_node_helper<N, JType>::print_remark("Recirculation test of tag-matching join");
        REMARK(" >\n");
        for(int maxTag = 1; maxTag <10; maxTag *= 3) {
            for(int i = 0; i < N; ++i) all_source_nodes[i][0] = NULL;

            tbb::flow::graph g;
            // this is the tag-matching join we're testing
            JType * my_join = makeJoin<N, JType, tbb::flow::tag_matching>::create(g);
            // source_node for continue messages
            tbb::flow::source_node<tbb::flow::continue_msg> snode(g, recirc_source_node_body(), false);
            // reserving join that matches recirculating tags with continue messages.
            input_join_type * my_input_join = makeJoin<2, input_join_type, tbb::flow::reserving>::create(g);
            // tbb::flow::make_edge(snode, tbb::flow::input_port<1>(*my_input_join));
            tbb::flow::make_edge(snode, tbb::flow::get<1>(my_input_join->input_ports()));
            // queue to hold the tags
            tbb::flow::queue_node<int> tag_queue(g);
            tbb::flow::make_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
            // add all the function_nodes that are inputs to the tag-matching join
            source_node_helper<N, JType>::add_recirc_func_nodes(*my_join, *my_input_join, g);
            // add the function_node that accepts the output of the join and emits the int tag it was based on
            tbb::flow::function_node<TType, int> recreate_tag(g, tbb::flow::unlimited, recirc_output_func_body<TType>());
            tbb::flow::make_edge(*my_join, recreate_tag);
            // now the recirculating part (output back to the queue)
            tbb::flow::make_edge(recreate_tag, tag_queue);

            // put the tags into the queue
            for(int t = 1; t<=maxTag; ++t) tag_queue.try_put(t);

            input_count = Recirc_count;
            output_count = 0;

            // start up the source node to get things going
            snode.activate();

            // wait for everything to stop
            g.wait_for_all();

            ASSERT(output_count==Recirc_count, "not all instances were received");

            int j;
            // grab the tags from the queue, record them
            std::vector<bool> out_tally(maxTag, false);
            for(int i = 0; i < maxTag; ++i) {
                ASSERT(tag_queue.try_get(j), "not enough tags in queue");
                ASSERT(!out_tally.at(j-1), "duplicate tag from queue");
                out_tally[j-1] = true;
            }
            ASSERT(!tag_queue.try_get(j), "Extra tags in recirculation queue");

            // deconstruct graph
            source_node_helper<N, JType>::remove_recirc_func_nodes(*my_join, *my_input_join);
            tbb::flow::remove_edge(*my_join, recreate_tag);
            makeJoin<N, JType, tbb::flow::tag_matching>::destroy(my_join);
            tbb::flow::remove_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
            tbb::flow::remove_edge(snode, tbb::flow::input_port<1>(*my_input_join));
            makeJoin<2, input_join_type, tbb::flow::reserving>::destroy(my_input_join);
        }
    }
};

template<typename JType>
class generate_recirc_test {
public:
    typedef tbb::flow::join_node<JType, tbb::flow::tag_matching> join_node_type;
    static void do_test() {
        tag_recirculation_test<join_node_type>::test();
    }
};

int TestMain() {
#if __TBB_USE_TBB_TUPLE
    REMARK("  Using TBB tuple\n");
#else
    REMARK("  Using platform tuple\n");
#endif

    TestTaggedBuffers();
    test_main<tbb::flow::queueing>();
    test_main<tbb::flow::reserving>();
    test_main<tbb::flow::tag_matching>();
    return Harness::Done;
}
