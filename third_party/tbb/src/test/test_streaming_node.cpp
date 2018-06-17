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

#if __TBB_PREVIEW_STREAMING_NODE

#if _MSC_VER
#pragma warning (disable: 4503) // Suppress "decorated name length exceeded, name was truncated" warning
#pragma warning (disable: 4702) // Suppress "unreachable code" warning
#endif

#include <functional>
#include <iostream>

#include "harness.h"
#include "harness_assert.h"

#include "tbb/concurrent_queue.h"
#include "tbb/flow_graph.h"
#include "tbb/tbb_thread.h"

using namespace tbb::flow;

//--------------------------------------------------------------------------------
//--------------------------------TEST HELPERS------------------------------------
//--------------------------------------------------------------------------------

template <typename ...A>
struct tuples_equal : std::false_type { };

template <typename ...A>
struct tuples_equal<std::tuple<A...>, std::tuple<>> : std::false_type { };

template <typename ...B>
struct tuples_equal<std::tuple<>, std::tuple<B...>> : std::false_type { };

template <>
struct tuples_equal<std::tuple<>, std::tuple<>> : std::true_type { };

template <typename A1, typename ...Aother, typename B1, typename ...Bother>
struct tuples_equal<std::tuple<A1, Aother...>, std::tuple<B1, Bother...>>
{
    static const bool value = std::is_same<A1, B1>::value && tuples_equal<std::tuple<Aother...>, std::tuple<Bother...>>::value;
};

template<typename...A>
struct first_variadic {
    template<typename...B>
    static void is_equal_to_second()
    {
        ASSERT((tuples_equal< std::tuple<A...>, std::tuple<B...> >::value), "Unexpected variadic types");
    }
};

//--------------------------------------------------------------------------------

template<typename T>
class factory_msg : public async_msg<T> {
public:
    factory_msg() {}
    factory_msg(const T& input_data) : m_data(input_data) {}

    const T& data() const { return m_data; }
    void update_data(T value) { m_data = value; }
private:
    T m_data;
};

//--------------------------------------------------------------------------------

class base_streaming_factory : NoCopy {
public:

    typedef int device_type;
    typedef int kernel_type;

    template<typename T> using async_msg_type = factory_msg<T>;

    base_streaming_factory() : devices_list(1) {}

    std::vector<device_type> devices() {
        return devices_list;
    }

    template <typename ...Args>
    void send_result_forward(Args&... args) {
        deviceResult = doDeviceWork();
        send_result(args...);
    }

    void clear_factory() {
        arguments_list.clear();
    }

    void process_arg_list() {}

    template <typename T, typename ...Rest>
    void process_arg_list(T& arg, Rest&... args) {
        process_one_arg(arg);
        process_arg_list(args...);
    }

private:

    int doDeviceWork() {
        int result = 0;
        for (size_t i = 0; i < arguments_list.size(); i++)
            result += arguments_list[i];
        return result;
    }

    // Pass calculation result to the next node
    template <typename ...Args>
    void set_result(Args...) {}

    template <typename T>
    void set_result(async_msg_type<T>& msg) {
        msg.set(deviceResult);
    }

    // Variadic functions for result processing
    // and sending them to all async_msgs
    void send_result() {}

    template <typename T, typename ...Rest>
    void send_result(T& arg, Rest&... args) {
        set_result(arg);
        send_result(args...);
    }

    // Retrieve values from async_msg objects
    // and store them in vector
    template <typename T>
    void process_one_arg(async_msg_type<T>& msg) {
        arguments_list.push_back(msg.data());
    }

    template <typename T>
    void process_one_arg(const async_msg_type<T>& msg) {
        arguments_list.push_back(msg.data());
    }

    std::vector<device_type> devices_list;
    std::vector<int> arguments_list;

    int deviceResult;
};

template<typename ...ExpectedArgs>
class test_streaming_factory : public base_streaming_factory {
public:

    template <typename ...Args>
    void send_data(device_type /*device*/, Args&... /*args*/) {}

    template <typename ...Args>
    void send_kernel(device_type /*device*/, const kernel_type& /*kernel*/, Args&... args) {
        check_arguments(args...);
        process_arg_list(args...);
        send_result_forward(args...);
        clear_factory();
    }

    template <typename FinalizeFn, typename ...Args>
    void finalize(device_type /*device*/, FinalizeFn fn, Args&... args) {
        check_arguments(args...);
        fn();
    }

    template<typename ...Args>
    void check_arguments(Args&... /*args*/) {
        first_variadic< Args... >::template is_equal_to_second< ExpectedArgs... >();
    }
};

//--------------------------------------------------------------------------------

template<typename Factory>
class device_selector {
public:
    device_selector() : my_state(DEFAULT_INITIALIZED) {}
    device_selector(const device_selector&) : my_state(COPY_INITIALIZED) {}
    device_selector(device_selector&&) : my_state(COPY_INITIALIZED) {}
    ~device_selector() { my_state = DELETED; }

    typename Factory::device_type operator()(Factory &f) {
        ASSERT(my_state == COPY_INITIALIZED, NULL);
        ASSERT(!f.devices().empty(), NULL);
        return *(f.devices().begin());
    }

private:
    enum state {
        DEFAULT_INITIALIZED,
        COPY_INITIALIZED,
        DELETED
    };
    state my_state;
};

//--------------------------------------------------------------------------------

void TestWithoutSetArgs() {
    graph g;

    typedef test_streaming_factory< factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);

    // test finalize function
    split_n.try_put(args_tuple);
    g.wait_for_all();

    make_edge(output_port<0>(streaming_n), function_n);
    expected_result = 30;
    split_n.try_put(args_tuple);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSetArgsOnly() {
    graph g;

    typedef test_streaming_factory< const factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);

    streaming_n.set_args(100);
    split_n.try_put(args_tuple);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSetPortRefOnly() {
    graph g;

    typedef test_streaming_factory< factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);


    streaming_n.set_args(port_ref<0, 1>());

    // test finalize function
    split_n.try_put(args_tuple);
    g.wait_for_all();

    make_edge(output_port<0>(streaming_n), function_n);
    expected_result = 30;
    split_n.try_put(args_tuple);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSetArgsAndPortRef1() {
    graph g;

    typedef test_streaming_factory< const factory_msg<int>, factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);

    streaming_n.set_args(100, port_ref<0, 1>());

    // test finalize function
    split_n.try_put(args_tuple);
    g.wait_for_all();

    make_edge(output_port<0>(streaming_n), function_n);
    expected_result = 130;
    split_n.try_put(args_tuple);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSetArgsAndPortRef2() {
    graph g;

    typedef test_streaming_factory< const factory_msg<int>, factory_msg<int>,
        const factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);

    streaming_n.set_args(100, port_ref<0>(), 200, port_ref<1>());

    // test finalize function
    split_n.try_put(args_tuple);
    g.wait_for_all();

    make_edge(output_port<0>(streaming_n), function_n);
    expected_result = 330;
    split_n.try_put(args_tuple);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

template <typename ...ExpectedArgs>
class send_data_factory : public base_streaming_factory {
public:

    send_data_factory() : send_data_counter(0) {}

    template <typename ...Args>
    void send_data(device_type /*device*/, Args&... /*args*/) {
        switch (send_data_counter) {
        case 0:
            first_variadic< Args... >::template is_equal_to_second< ExpectedArgs... >();
            break;
        case 1:
            first_variadic< Args... >::template is_equal_to_second< factory_msg<int> >();
            break;
        case 2:
            first_variadic< Args... >::template is_equal_to_second< factory_msg<int> >();
            break;
        default:
            break;
        }
        send_data_counter++;
    }

    template <typename ...Args>
    void send_kernel(device_type /*device*/, const kernel_type& /*kernel*/, Args&... /*args*/) {
        ASSERT(send_data_counter == 3, "send_data() was called not enough times");
        send_data_counter = 0;
    }

    template <typename FinalizeFn, typename ...Args>
    void finalize(device_type /*device*/, FinalizeFn fn, Args&... /*args*/) {
        fn();
    }

private:
    int send_data_counter;
};

void TestSendData_withoutSetArgs() {
    graph g;

    typedef send_data_factory< tbb::flow::interface10::internal::port_ref_impl<0, 1> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    input_port<0>(streaming_n).try_put(10);
    input_port<1>(streaming_n).try_put(20);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSendData_setArgsOnly() {
    graph g;

    typedef send_data_factory< factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    streaming_n.set_args(100);
    input_port<0>(streaming_n).try_put(10);
    input_port<1>(streaming_n).try_put(20);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSendData_portRefOnly() {
    graph g;

    typedef send_data_factory< tbb::flow::interface10::internal::port_ref_impl<0,1> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    streaming_n.set_args(port_ref<0,1>());
    input_port<0>(streaming_n).try_put(10);
    input_port<1>(streaming_n).try_put(20);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSendData_setArgsAndPortRef1() {
    graph g;

    typedef send_data_factory< factory_msg<int>, tbb::flow::interface10::internal::port_ref_impl<0, 1> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    streaming_n.set_args(100, port_ref<0,1>());
    input_port<0>(streaming_n).try_put(10);
    input_port<1>(streaming_n).try_put(20);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestSendData_setArgsAndPortRef2() {
    graph g;

    typedef send_data_factory< factory_msg<int>, tbb::flow::interface10::internal::port_ref_impl<0,0>,
                               factory_msg<int>, tbb::flow::interface10::internal::port_ref_impl<1,1> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    streaming_n.set_args(100, port_ref<0>(), 200, port_ref<1>());
    input_port<0>(streaming_n).try_put(10);
    input_port<1>(streaming_n).try_put(20);
    g.wait_for_all();
}

//--------------------------------------------------------------------------------

void TestArgumentsPassing() {
    REMARK("TestArgumentsPassing: ");
    TestWithoutSetArgs();
    TestSetArgsOnly();
    TestSetPortRefOnly();
    TestSetArgsAndPortRef1();
    TestSetArgsAndPortRef2();

    TestSendData_withoutSetArgs();
    TestSendData_setArgsOnly();
    TestSendData_portRefOnly();
    TestSendData_setArgsAndPortRef1();
    TestSendData_setArgsAndPortRef2();
    REMARK("done\n");
}

//--------------------------------------------------------------------------------

template<typename... ExpectedArgs>
class range_streaming_factory : public base_streaming_factory {
public:

    typedef std::array<int, 2> range_type;

    template <typename ...Args>
    void send_data(device_type /*device*/, Args&... /*args*/) {
    }

    template <typename ...Args>
    void send_kernel(device_type /*device*/, const kernel_type& /*kernel*/, const range_type& work_size, Args&... args) {
        ASSERT(work_size[0] == 1024, "Range was set incorrectly");
        ASSERT(work_size[1] == 720, "Range was set incorrectly");
        first_variadic< Args... >::template is_equal_to_second< ExpectedArgs... >();
        process_arg_list(args...);
        send_result_forward(args...);
        clear_factory();
    }

    template <typename FinalizeFn, typename ...Args>
    void finalize(device_type /*device*/, FinalizeFn fn, Args&... /*args*/) {
        first_variadic< Args... >::template is_equal_to_second< ExpectedArgs... >();
        fn();
    }

};

void TestSetRange() {
    REMARK("TestSetRange: ");

    graph g;

    typedef range_streaming_factory< const factory_msg<int>, factory_msg<int>,
        const factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n));

    const int first_arg = 10;
    const int second_arg = 20;
    std::tuple<int, int> args_tuple = std::make_tuple(first_arg, second_arg);

    streaming_n.set_args(100, port_ref<0>(), 200, port_ref<1>());

    // test version for GCC <= 4.7.2 (unsupported conversion from initializer_list to std::array)
#if __GNUC__ < 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ <= 7 || (__GNUC_MINOR__ == 7 && __GNUC_PATCHLEVEL__ <= 2)))
    std::array<int, 2> device_range;
    device_range[0] = 1024;
    device_range[1] = 720;
    streaming_n.set_range(device_range);
#else
    std::array<int, 2> device_range = { 1024,720 };
    streaming_n.set_range(device_range);
#endif

    split_n.try_put(args_tuple);
    g.wait_for_all();

    make_edge(output_port<0>(streaming_n), function_n);
    expected_result = 330;
    split_n.try_put(args_tuple);
    g.wait_for_all();

    REMARK("done\n");
}

//-------------------------------------------------------------------------------------------------------------------------------------------

template <typename T>
class user_async_msg : public tbb::flow::async_msg<T>
{
public:
    user_async_msg() {}
    user_async_msg(T value) : m_data(value) {}
    void finalize() const __TBB_override;
private:
    T m_data;
};

class user_async_activity { // Async activity singleton
public:

    static user_async_activity* instance() {
        if (s_Activity == NULL) {
            s_Activity = new user_async_activity();
        }
        return s_Activity;
    }

    static void destroy() {
        ASSERT(s_Activity != NULL, "destroyed twice");
        s_Activity->myThread.join();
        delete s_Activity;
        s_Activity = NULL;
    }

    template <typename FinalizeFn>
    static void finish(FinalizeFn fn) {
        ASSERT(user_async_activity::s_Activity != NULL, "activity must be alive");
        user_async_activity::s_Activity->finishTaskQueue(fn);
    }

    static void finish(const user_async_msg<int>& msg) {
        ASSERT(user_async_activity::s_Activity != NULL, "activity must be alive");
        user_async_activity::s_Activity->finishTaskQueue(msg);
    }

    static int getResult() {
        ASSERT(user_async_activity::s_Activity != NULL, "activity must be alive");
        return user_async_activity::s_Activity->myQueueSum;
    }

    void addWork(int addValue, int timeout = 0) {
        myQueue.push(my_task(addValue, timeout));
    }

    template <typename FinalizeFn>
    void finishTaskQueue(FinalizeFn fn) {
        myFinalizer = fn;
        myQueue.push(my_task(0, 0, true));
    }

    void finishTaskQueue(const user_async_msg<int>& msg) {
        myMsg = msg;
        myQueue.push(my_task(0, 0, true));
    }

private:

    struct my_task {
        my_task(int addValue = 0, int timeout = 0, bool finishFlag = false)
            : myAddValue(addValue), myTimeout(timeout), myFinishFlag(finishFlag) {}

        int     myAddValue;
        int     myTimeout;
        bool    myFinishFlag;
    };

    static void threadFunc(user_async_activity* activity) {
        for (;;) {
            my_task work;
            activity->myQueue.pop(work);
            Harness::Sleep(work.myTimeout);
            if (work.myFinishFlag) {
                break;
            }
            activity->myQueueSum += work.myAddValue;
        }

        // Send result back to the graph
        if (activity->myFinalizer) {
            activity->myFinalizer();
        }
        activity->myMsg.set(activity->myQueueSum);

    }

    user_async_activity() : myQueueSum(0), myThread(&user_async_activity::threadFunc, this) {}

    tbb::concurrent_bounded_queue<my_task>   myQueue;
    int                                      myQueueSum;
    user_async_msg<int>                      myMsg;
    std::function<void(void)>                myFinalizer;
    tbb::tbb_thread                          myThread;

    static user_async_activity*              s_Activity;
};

user_async_activity* user_async_activity::s_Activity = NULL;

template <typename T>
void user_async_msg<T>::finalize() const {
    user_async_activity::finish(*this);
}

class data_streaming_factory {
public:

    typedef int device_type;
    typedef int kernel_type;

    template<typename T> using async_msg_type = user_async_msg<T>;

    data_streaming_factory() : devices_list(1) {}

    template <typename ...Args>
    void send_data(device_type /*device*/, Args&... /*args*/) {}

    template <typename ...Args>
    void send_kernel(device_type /*device*/, const kernel_type& /*kernel*/, Args&... args) {
        process_arg_list(args...);
    }

    template <typename FinalizeFn, typename ...Args>
    void finalize(device_type /*device*/, FinalizeFn fn, Args&... /*args*/) {
        user_async_activity::finish(fn);
    }

    // Retrieve values from async_msg objects
    // and store them in vector
    void process_arg_list() {}

    template <typename T, typename ...Rest>
    void process_arg_list(T& arg, Rest&... args) {
        process_one_arg(arg);
        process_arg_list(args...);
    }

    template <typename T>
    void process_one_arg(async_msg_type<T>& /*msg*/) {
        user_async_activity::instance()->addWork(1, 10);
    }

    template <typename ...Args>
    void process_one_arg(Args&... /*args*/) {}

    std::vector<device_type> devices() {
        return devices_list;
    }

private:
    std::vector<device_type> devices_list;
};

void TestChaining() {
    REMARK("TestChaining: ");

    graph g;
    typedef streaming_node< tuple<int>, queueing, data_streaming_factory > streaming_node_type;
    typedef std::vector< streaming_node_type > nodes_vector_type;

    data_streaming_factory factory;
    device_selector<data_streaming_factory> device_selector;
    data_streaming_factory::kernel_type kernel(0);

    const int STREAMING_GRAPH_CHAIN_LENGTH = 1000;
    nodes_vector_type nodes_vector;
    for (int i = 0; i < STREAMING_GRAPH_CHAIN_LENGTH; i++) {
        nodes_vector.emplace_back(g, kernel, device_selector, factory);
    }

    function_node< int, int > source_n(g, unlimited, [&g](const int& value) -> int {
        return value;
    });

    function_node< int > destination_n(g, unlimited, [&g, &STREAMING_GRAPH_CHAIN_LENGTH](const int& result) {
        ASSERT(result == STREAMING_GRAPH_CHAIN_LENGTH, "calculation chain result is wrong");
    });

    make_edge(source_n, input_port<0>(nodes_vector.front()));
    for (size_t i = 0; i < nodes_vector.size() - 1; i++) {
        make_edge(output_port<0>(nodes_vector[i]), input_port<0>(nodes_vector[i + 1]));
        nodes_vector[i].set_args(port_ref<0>());
    }
    nodes_vector.back().set_args(port_ref<0>());
    make_edge(output_port<0>(nodes_vector.back()), destination_n);

    source_n.try_put(0);
    g.wait_for_all();

    REMARK("result = %d; expected = %d\n", user_async_activity::getResult(), STREAMING_GRAPH_CHAIN_LENGTH);
    ASSERT(user_async_activity::getResult() == STREAMING_GRAPH_CHAIN_LENGTH, "calculation chain result is wrong");

    user_async_activity::destroy();

    REMARK("done\n");
}

//--------------------------------------------------------------------------------

void TestCopyConstructor() {
    REMARK("TestCopyConstructor: ");

    graph g;

    typedef test_streaming_factory< factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    // Testing copy constructor
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n_copied(streaming_n);

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n_copied));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n_copied));
    make_edge(output_port<0>(streaming_n_copied), function_n);

    std::tuple<int, int> args_tuple = std::make_tuple(10, 20);
    expected_result = 30;
    split_n.try_put(args_tuple);
    g.wait_for_all();

    REMARK("done\n");
}

void TestMoveConstructor() {
    REMARK("TestMoveConstructor: ");

    graph g;

    typedef test_streaming_factory< factory_msg<int>, factory_msg<int> > device_factory;

    device_factory factory;
    device_selector<device_factory> device_selector;
    device_factory::kernel_type kernel(0);

    int expected_result;
    split_node < tuple<int, int> > split_n(g);
    function_node< int > function_n(g, unlimited, [&expected_result](const int& result) {
        ASSERT(expected_result == result, "Validation has failed");
    });

    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n(g, kernel, device_selector, factory);

    // Testing move constructor
    streaming_node< tuple<int, int>, queueing, device_factory > streaming_n_moved(std::move(streaming_n));

    make_edge(output_port<0>(split_n), input_port<0>(streaming_n_moved));
    make_edge(output_port<1>(split_n), input_port<1>(streaming_n_moved));
    make_edge(output_port<0>(streaming_n_moved), function_n);

    std::tuple<int, int> args_tuple = std::make_tuple(10, 20);
    expected_result = 30;
    split_n.try_put(args_tuple);
    g.wait_for_all();

    REMARK("done\n");
}

void TestConstructor() {
    TestCopyConstructor();
    TestMoveConstructor();
}

//--------------------------------------------------------------------------------

int TestMain() {
    TestArgumentsPassing();
    TestSetRange();
    TestChaining();
    TestConstructor();
    return Harness::Done;
}
#else
#define HARNESS_SKIP_TEST 1
#include "harness.h"
#endif
