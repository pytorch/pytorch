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

#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

#include "tbb/tbb_config.h"

#if __TBB_PREVIEW_GFX_FACTORY && __TBB_PREVIEW_STREAMING_NODE

#if _MSC_VER
#pragma warning (disable: 4503) // Suppress "decorated name length exceeded, name was truncated" warning
#endif

#include "tbb/flow_graph.h"
#include "tbb/gfx_factory.h"

#include <cilk/cilk.h>

#include "harness.h"
#include "harness_assert.h"

using namespace tbb::flow;

//---------------------------------------------------------------------------------------------------------------------------------
// Helpers
//---------------------------------------------------------------------------------------------------------------------------------

typedef tuple< gfx_buffer<int>, size_t > kernel_args;
typedef streaming_node< kernel_args, queueing, gfx_factory > gfx_node;

template <typename T>
void init_random_buffer(gfx_buffer<T>& buf) {
    Harness::FastRandom rnd(42);
    std::generate(buf.begin(), buf.end(), [&rnd]() { return rnd.get(); });
}

template <typename T>
void copy_buffer_to_vector(gfx_buffer<T>& buf, std::vector<T>& vect) {
    std::copy(buf.begin(), buf.end(), std::back_inserter(vect));
}

//---------------------------------------------------------------------------------------------------------------------------------

// GFX functions to offload
static __declspec(target(gfx_kernel))
void sq_vec(int *v, size_t n) {
    cilk_for(size_t i = 0; i < n; ++i) {
        v[i] = v[i] * v[i];
    }
}

// Reference function
void sq_vec_ref(std::vector<int>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = v[i] * v[i];
    }
}

//---------------------------------------------------------------------------------------------------------------------------------

void TestDynamicKernelArgs_finalize() {
    REMARK("    TestDynamicKernelArgs_finalize: ");

    // Initialize input data
    const size_t array_size = 1000;
    gfx_buffer<int> buffer(array_size);

    // Generate random buffer values
    init_random_buffer(buffer);

    // Copy buffer to vector for the next validation
    std::vector<int> check_vec;
    copy_buffer_to_vector(buffer, check_vec);

    // Obtain reference result
    sq_vec_ref(check_vec);

    graph g;
    gfx_factory factory(g);

    gfx_node streaming_n(g, sq_vec, gfx_factory::dummy_device_selector(), factory);

    streaming_n.set_args(port_ref<0, 1>);
    input_port<0>(streaming_n).try_put(buffer);
    input_port<1>(streaming_n).try_put(array_size);

    g.wait_for_all();

    ASSERT((buffer.size() == check_vec.size()), "Validation has failed");
    ASSERT((std::equal(buffer.begin(), buffer.end(), check_vec.begin())), "Validation has failed");
    REMARK("done\n");
}

void TestConstantKernelArgs_finalize() {
    REMARK("    TestConstantKernelArgs_finalize: ");

    // Initialize input data
    const size_t array_size = 1000;
    gfx_buffer<int> buffer(array_size);

    // Generate random buffer values
    init_random_buffer(buffer);

    // Copy buffer to vector for the next validation
    std::vector<int> check_vec;
    copy_buffer_to_vector(buffer, check_vec);

    // Obtain reference result
    sq_vec_ref(check_vec);

    graph g;
    gfx_factory factory(g);

    streaming_node< tuple< gfx_buffer<int> >, queueing, gfx_factory > streaming_n(g, sq_vec, gfx_factory::dummy_device_selector(), factory);

    streaming_n.set_args(port_ref<0>(), array_size);
    input_port<0>(streaming_n).try_put(buffer);

    g.wait_for_all();

    ASSERT((buffer.size() == check_vec.size()), "Validation has failed");
    ASSERT((std::equal(buffer.begin(), buffer.end(), check_vec.begin())), "Validation has failed");

    REMARK("done\n");
}

void TestGfxStreamingFactory_finalize() {
    REMARK("TestGfxStreamingFactory_finalize: ");
    TestDynamicKernelArgs_finalize();
    TestConstantKernelArgs_finalize();
    REMARK("done\n");
}

//---------------------------------------------------------------------------------------------------------------------------------

void TestDynamicKernelArgs_send_kernel() {
    REMARK("    TestDynamicKernelArgs_send_kernel: ");

    // Initialize input data
    const size_t array_size = 1000;
    gfx_buffer<int> buffer(array_size);

    // Generate random buffer values
    init_random_buffer(buffer);

    // Copy buffer to vector for the next validation
    std::vector<int> check_vec;
    copy_buffer_to_vector(buffer, check_vec);

    // Obtain reference result
    sq_vec_ref(check_vec);

    graph g;
    gfx_factory factory(g);

    gfx_node streaming_n(g, sq_vec, gfx_factory::dummy_device_selector(), factory);

    join_node< kernel_args > join_n(g);
    function_node< kernel_args > function_n(g, unlimited, [&check_vec](const kernel_args& result) {
        gfx_buffer<int> buffer = get<0>(result);

        ASSERT((buffer.size() == check_vec.size()), "Validation has failed");
        ASSERT((std::equal(buffer.begin(), buffer.end(), check_vec.begin())), "Validation has failed");
    });

    make_edge(output_port<0>(streaming_n), input_port<0>(join_n));
    make_edge(output_port<1>(streaming_n), input_port<1>(join_n));
    make_edge(join_n, function_n);

    streaming_n.set_args(port_ref<0, 1>);
    input_port<0>(streaming_n).try_put(buffer);
    input_port<1>(streaming_n).try_put(array_size);

    g.wait_for_all();

    REMARK("done\n");
}

void TestConstantKernelArgs_send_kernel() {
    REMARK("    TestConstantKernelArgs_send_kernel: ");

    // Initialize input data
    const size_t array_size = 1000;
    gfx_buffer<int> buffer(array_size);

    // Generate random buffer values
    init_random_buffer(buffer);

    // Copy buffer to vector for the next validation
    std::vector<int> check_vec;
    copy_buffer_to_vector(buffer, check_vec);

    // Obtain reference result
    sq_vec_ref(check_vec);

    graph g;
    gfx_factory factory(g);

    streaming_node< tuple< gfx_buffer<int> >, queueing, gfx_factory > streaming_n(g, sq_vec, gfx_factory::dummy_device_selector(), factory);

    join_node< tuple< gfx_buffer<int> > > join_n(g);
    function_node< tuple< gfx_buffer<int> > > function_n(g, unlimited, [&check_vec](const tuple< gfx_buffer<int> >& result) {
        gfx_buffer<int> buffer = get<0>(result);

        ASSERT((buffer.size() == check_vec.size()), "Validation has failed");
        ASSERT((std::equal(buffer.begin(), buffer.end(), check_vec.begin())), "Validation has failed");
    });

    make_edge(output_port<0>(streaming_n), input_port<0>(join_n));
    make_edge(join_n, function_n);

    streaming_n.set_args(port_ref<0>(), array_size);
    input_port<0>(streaming_n).try_put(buffer);

    g.wait_for_all();

    REMARK("done\n");
}

void TestGfxStreamingFactory_send_kernel() {
    REMARK("TestGfxStreamingFactory_send_kernel:\n");
    TestDynamicKernelArgs_send_kernel();
    TestConstantKernelArgs_send_kernel();
    REMARK("done\n");
}

//---------------------------------------------------------------------------------------------------------------------------------

void ConcurrencyTest() {
    REMARK("ConcurrencyTest: ");

    // Initialize input data
    const size_t array_size = 1000;
    gfx_buffer<int> buffer(array_size);

    // Generate random buffer values
    init_random_buffer(buffer);

    // Copy buffer to vector for the next validation
    std::vector<int> check_vec;
    copy_buffer_to_vector(buffer, check_vec);

    // Obtain reference result
    sq_vec_ref(check_vec);

    graph g;
    gfx_factory factory(g);

    streaming_node< tuple< gfx_buffer<int> >, queueing, gfx_factory > streaming_n(g, sq_vec, gfx_factory::dummy_device_selector(), factory);

    join_node< tuple< gfx_buffer<int> > > join_n(g);
    function_node< tuple< gfx_buffer<int> > > function_n(g, unlimited, [&check_vec](const tuple< gfx_buffer<int> >& result) {
        gfx_buffer<int> buffer = get<0>(result);

        ASSERT((buffer.size() == check_vec.size()), "Validation has failed");
        ASSERT((std::equal(buffer.begin(), buffer.end(), check_vec.begin())), "Validation has failed");
    });

    make_edge(output_port<0>(streaming_n), input_port<0>(join_n));
    make_edge(join_n, function_n);

    streaming_n.set_args(port_ref<0>(), array_size);

    for (int i = 0; i < 100; i++) {
        gfx_buffer<int> input(array_size);

        for (int i = 0; i < buffer.size(); i++) {
            input[i] = buffer[i];
        }

        input_port<0>(streaming_n).try_put(input);
    }

    g.wait_for_all();

    REMARK("done\n");
}

//---------------------------------------------------------------------------------------------------------------------------------

int TestMain() {
    TestGfxStreamingFactory_finalize();
    TestGfxStreamingFactory_send_kernel();
    ConcurrencyTest();
    return Harness::Done;
}

#else
#define HARNESS_SKIP_TEST 1
#include "harness.h"
#endif
