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

#include <iostream>
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

static double bm_split_node(tbb::flow::graph& g, int nIter);
static double bm_broadcast_node(tbb::flow::graph& g, int nIter);
static double bm_queue_node(tbb::flow::graph& g, int nIter);

typedef int my_type;
//typedef std::vector<int> my_type;

const int nIter = 1 << 24; //16M
const int nSize = 100000000;

int main()
{
    //set up one thread to eliminate scheduler overheads
    tbb::task_scheduler_init tsi(1);

    tbb::flow::graph g;

    //1. queue_node benchmark; calculate queue_node time + plus threads creation time (if we have multi-threading)
    std::cout << "queue benchmark: number of calls of putting element:" << nIter;
    const double tQueue = bm_queue_node(g, nIter);
    std::cout << ";  time:" << tQueue << std::endl << std::endl;

    //2. split_node benchmark
    std::cout << "split_node benchmark: number of calls:" << nIter;
    const double tSplitNode = bm_split_node(g, nIter);
    //output split_node benchmark result
    std::cout << ";  time:" << tSplitNode << std::endl;
    std::cout << "exclusive split_node time:" << tSplitNode - tQueue << std::endl << std::endl;

    //3. broadcast_node benchmark
    std::cout << "broadcast_node benchmark: number of calls:" << nIter;
    const double tBNode = bm_broadcast_node(g, nIter);
    //output broadcast_node benchmark result
    std::cout << ";  time:" << tBNode << std::endl;
    std::cout << "exclusive broadcast_node time:" << tBNode - tQueue << std::endl;

    return 0;
}

//! Dummy executing split_node, "nIter" calls; Returns time in seconds.
double bm_split_node(tbb::flow::graph& g, int nIter)
{
    my_type v1(nSize);

    tbb::flow::queue_node<my_type> my_queue1(g);
    tbb::flow::tuple<my_type> my_tuple(1);

    tbb::flow::split_node< tbb::flow::tuple<my_type> > my_split_node(g);
    make_edge(tbb::flow::get<0>(my_split_node.output_ports()), my_queue1);

    const tbb::tick_count t0 = tbb::tick_count::now();

    //using split_node
    for (int i = 0; i < nIter; ++i)
        my_split_node.try_put(my_tuple); 

    //barrier sync
    g.wait_for_all();

    return (tbb::tick_count::now() - t0).seconds();
}

//! Dummy executing broadcast_node; "nIter" calls; Returns time in seconds.
double bm_broadcast_node(tbb::flow::graph& g, int nIter)
{
    tbb::flow::queue_node<my_type> my_queue(g);
    tbb::flow::broadcast_node<my_type> my_broadcast_node(g);
    make_edge(my_broadcast_node, my_queue);

    my_type v(nSize);

    const tbb::tick_count t0 = tbb::tick_count::now();

    //using broadcast_node
    for (int i = 0; i < nIter; ++i)
        my_broadcast_node.try_put(v);
    //barrier sync
    g.wait_for_all();

    return (tbb::tick_count::now() - t0).seconds();
}

double bm_queue_node(tbb::flow::graph& g, int nIter)
{
    tbb::flow::queue_node<my_type> first_queue(g);

    my_type v(nSize);

    tbb::tick_count t0 = tbb::tick_count::now();
    //using queue_node
    for (int i = 0; i < nIter; ++i)
        first_queue.try_put(v);
    g.wait_for_all();
    return (tbb::tick_count::now() - t0).seconds();
}
