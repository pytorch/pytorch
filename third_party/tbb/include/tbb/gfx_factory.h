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

#ifndef __TBB_flow_graph_gfx_factory_H
#define __TBB_flow_graph_gfx_factory_H

#include "tbb/tbb_config.h"

#if __TBB_PREVIEW_GFX_FACTORY

#include <vector>
#include <future>
#include <mutex>
#include <iostream>

#include <gfx/gfx_rt.h>
#include <gfx/gfx_intrin.h>
#include <gfx/gfx_types.h>

namespace tbb {

namespace flow {

namespace interface9 {

template <typename T>
class gfx_buffer;

namespace gfx_offload {

    typedef GfxTaskId task_id_type;

    //-----------------------------------------------------------------------
    // GFX errors checkers.
    // For more debug output, set GFX_LOG_OFFLOAD=2 macro
    //-----------------------------------------------------------------------

    // TODO: reconsider error handling approach. If exception is the right way
    // then need to define and document a specific exception type.
    inline void throw_gfx_exception() {
        std::string msg = "GFX error occurred: " + std::to_string(_GFX_get_last_error());
        std::cerr << msg << std::endl;
        throw msg;
    }

    inline void check_enqueue_retcode(task_id_type err) {
        if (err == 0) {
            throw_gfx_exception();
        }
    }

    inline void check_gfx_retcode(task_id_type err) {
        if (err != GFX_SUCCESS) {
            throw_gfx_exception();
        }
    }

    //---------------------------------------------------------------------
    // GFX asynchronous offload and share API
    //---------------------------------------------------------------------

    // Sharing and unsharing data API
    template<typename DataType, typename SizeType>
    void share(DataType* p, SizeType n) { check_gfx_retcode(_GFX_share(p, sizeof(*p)*n)); }
    template<typename DataType>
    void unshare(DataType* p) { check_gfx_retcode(_GFX_unshare(p)); }

    // Retrieving array pointer from shared gfx_buffer
    // Other types remain the same
    template <typename T>
    T* raw_data(gfx_buffer<T>& buffer) { return buffer.data(); }
    template <typename T>
    const T* raw_data(const gfx_buffer<T>& buffer) { return buffer.data(); }
    template <typename T>
    T& raw_data(T& data) { return data; }
    template <typename T>
    const T& raw_data(const T& data) { return data; }

    // Kernel enqueuing on device with arguments
    template <typename F, typename ...ArgType>
    task_id_type run_kernel(F ptr, ArgType&... args) {
        task_id_type id = _GFX_offload(ptr, raw_data(args)...);

        // Check if something during offload went wrong (ex: driver initialization failure)
        gfx_offload::check_enqueue_retcode(id);

        return id;
    }

    // Waiting for tasks completion
    void wait_for_task(task_id_type id) { check_gfx_retcode(_GFX_wait(id)); }

} // namespace gfx_offload

template <typename T>
class gfx_buffer {
public:

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    typedef std::size_t size_type;

    gfx_buffer() : my_vector_ptr(std::make_shared< std::vector<T> >()) {}
    gfx_buffer(size_type size) : my_vector_ptr(std::make_shared< std::vector<T> >(size)) {}

    T* data() { return &(my_vector_ptr->front()); }
    const T* data() const { return &(my_vector_ptr->front()); }

    size_type size() const { return my_vector_ptr->size(); }

    const_iterator cbegin() const { return my_vector_ptr->cbegin(); }
    const_iterator cend() const { return my_vector_ptr->cend(); }
    iterator begin() { return my_vector_ptr->begin(); }
    iterator end() { return my_vector_ptr->end(); }

    T& operator[](size_type pos) { return (*my_vector_ptr)[pos]; }
    const T& operator[](size_type pos) const { return (*my_vector_ptr)[pos]; }

private:
    std::shared_ptr< std::vector<T> > my_vector_ptr;
};

template<typename T>
class gfx_async_msg : public tbb::flow::async_msg<T> {
public:
    typedef gfx_offload::task_id_type kernel_id_type;

    gfx_async_msg() : my_task_id(0) {}
    gfx_async_msg(const T& input_data) : my_data(input_data), my_task_id(0) {}

    T& data() { return my_data; }
    const T& data() const { return my_data; }

    void set_task_id(kernel_id_type id) { my_task_id = id; }
    kernel_id_type task_id() const { return my_task_id; }

private:
    T my_data;
    kernel_id_type my_task_id;
};

class gfx_factory {
private:

    // Wrapper for GFX kernel which is just a function
    class func_wrapper {
    public:

        template <typename F>
        func_wrapper(F ptr) { my_ptr = reinterpret_cast<void*>(ptr); }

        template<typename ...Args>
        void operator()(Args&&... args) {}

        operator void*() { return my_ptr; }

    private:
        void* my_ptr;
    };

public:

    // Device specific types
    template<typename T> using async_msg_type = gfx_async_msg<T>;

    typedef func_wrapper kernel_type;

    // Empty device type that is needed for Factory Concept
    // but is not used in gfx_factory
    typedef struct {} device_type;

    typedef gfx_offload::task_id_type kernel_id_type;

    gfx_factory(tbb::flow::graph& g) : m_graph(g), current_task_id(0) {}

    // Upload data to the device
    template <typename ...Args>
    void send_data(device_type /*device*/, Args&... args) {
        send_data_impl(args...);
    }

    // Run kernel on the device
    template <typename ...Args>
    void send_kernel(device_type /*device*/, const kernel_type& kernel, Args&... args) {
        // Get packed T data from async_msg<T> and pass it to kernel
        kernel_id_type id = gfx_offload::run_kernel(kernel, args.data()...);

        // Set id to async_msg
        set_kernel_id(id, args...);

        // Extend the graph lifetime until the callback completion.
        m_graph.reserve_wait();

        // Mutex for future assignment
        std::lock_guard<std::mutex> lock(future_assignment_mutex);

        // Set callback that waits for kernel execution
        callback_future = std::async(std::launch::async, &gfx_factory::callback<Args...>, this, id, args...);
    }

    // Finalization action after the kernel run
    template <typename FinalizeFn, typename ...Args>
    void finalize(device_type /*device*/, FinalizeFn fn, Args&... /*args*/) {
        fn();
    }

    // Empty device selector.
    // No way to choose a device with GFX API.
    class dummy_device_selector {
    public:
        device_type operator()(gfx_factory& /*factory*/) {
            return device_type();
        }
    };

private:

    //---------------------------------------------------------------------
    // Callback for kernel result
    //---------------------------------------------------------------------

    template <typename ...Args>
    void callback(kernel_id_type id, Args... args) {
        // Waiting for specific tasks id to complete
        {
            std::lock_guard<std::mutex> lock(task_wait_mutex);
            if (current_task_id < id) {
                gfx_offload::wait_for_task(id);
                current_task_id = id;
            }
        }

        // Get result from device and set to async_msg (args)
        receive_data(args...);

        // Data was sent to the graph, release the reference
        m_graph.release_wait();
    }

    //---------------------------------------------------------------------
    // send_data() arguments processing
    //---------------------------------------------------------------------

    // GFX buffer shared data with device that will be executed on
    template <typename T>
    void share_data(T) {}

    template <typename T>
    void share_data(gfx_buffer<T>& buffer) {
        gfx_offload::share(buffer.data(), buffer.size());
    }

    template <typename T>
    void send_arg(T) {}

    template <typename T>
    void send_arg(async_msg_type<T>& msg) {
        share_data(msg.data());
    }

    void send_data_impl() {}

    template <typename T, typename ...Rest>
    void send_data_impl(T& arg, Rest&... args) {
        send_arg(arg);
        send_data_impl(args...);
    }

    //----------------------------------------------------------------------
    // send_kernel() arguments processing
    //----------------------------------------------------------------------

    template <typename T>
    void set_kernel_id_arg(kernel_id_type, T) {}

    template <typename T>
    void set_kernel_id_arg(kernel_id_type id, async_msg_type<T>& msg) {
        msg.set_task_id(id);
    }

    void set_kernel_id(kernel_id_type) {}

    template <typename T, typename ...Rest>
    void set_kernel_id(kernel_id_type id, T& arg, Rest&... args) {
        set_kernel_id_arg(id, arg);
        set_kernel_id(id, args...);
    }

    //-----------------------------------------------------------------------
    // Arguments processing after kernel execution.
    // Unsharing buffers and forwarding results to the graph
    //-----------------------------------------------------------------------

    // After kernel execution the data should be unshared
    template <typename T>
    void unshare_data(T) {}

    template <typename T>
    void unshare_data(gfx_buffer<T>& buffer) {
        gfx_offload::unshare(buffer.data());
    }

    template <typename T>
    void receive_arg(T) {}

    template <typename T>
    void receive_arg(async_msg_type<T>& msg) {
        unshare_data(msg.data());
        msg.set(msg.data());
    }

    void receive_data() {}

    template <typename T, typename ...Rest>
    void receive_data(T& arg, Rest&... args) {
        receive_arg(arg);
        receive_data(args...);
    }

    //-----------------------------------------------------------------------
    int current_task_id;

    std::future<void> callback_future;
    tbb::flow::graph& m_graph;

    std::mutex future_assignment_mutex;
    std::mutex task_wait_mutex;
};

} // namespace interface9

using interface9::gfx_factory;
using interface9::gfx_buffer;

} // namespace flow

} // namespace tbb

#endif // __TBB_PREVIEW_GFX_FACTORY

#endif // __TBB_flow_graph_gfx_factory_H
